"""Negamax search: alpha-beta, ID, aspiration, NMP, IID, killers, history, LMR, futility, SE."""

import math
import time
from typing import cast

import chess
import numpy as np

from . import tt as _tt
from .bitboard import encode, evaluate
from .movegen import (
    bishop_attacks_bb,
    generate_pseudo_legal,
    king_attacks_bb,
    knight_attacks_bb,
    pawn_attacks_bb,
    queen_attacks_bb,
    rook_attacks_bb,
)
from .time_manager import TimeManager
from .tt import Flag, TTEntry

MATE: int = 1_000_000
_MATE_THRESHOLD: int = MATE // 2
_MAX_DEPTH = 50
_ASPIRATION_DELTA = 50
_IID_MIN_DEPTH = 3
_NMP_R = 2
_LMR_MIN_DEPTH = 3
_LMR_MIN_MOVE_INDEX = 4
_SE_MIN_DEPTH = 6
_SE_MARGIN = 50.0
# Futility: at depth 1 only, skip quiet non-special moves when eval is far below alpha.
_FUTILITY_MARGIN = 150       # centipawns — one pawn and a bit of tempo
_DELTA_MARGIN = 200          # quiescence delta pruning buffer (promotion bonus)
_MAX_GAIN = 900 + _DELTA_MARGIN  # = 1100; queen value + buffer
_INSTABILITY_THRESHOLD = 30.0  # cp swing between depths that triggers a time extension
_INSTABILITY_FACTOR = 1.5


def _tt_norm(score: float, ply: int) -> float:
    """Convert node-relative mate score to root-relative before TT storage."""
    if score > _MATE_THRESHOLD:
        return score + ply
    if score < -_MATE_THRESHOLD:
        return score - ply
    return score


def _tt_denorm(score: float, ply: int) -> float:
    """Convert root-relative mate score back to node-relative after TT retrieval."""
    if score > _MATE_THRESHOLD:
        return score - ply
    if score < -_MATE_THRESHOLD:
        return score + ply
    return score


_PIECE_VAL: dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

# Two killer moves per ply; cleared each get_move call.
_killers: list[list[chess.Move | None]] = cast(
    list[list[chess.Move | None]],
    [[None, None] for _ in range(_MAX_DEPTH + 2)],
)
_NO_KILLERS: list[chess.Move | None] = [None, None]

# History heuristic: _history[from_sq][to_sq] accumulates depth^2 on quiet beta cutoffs.
# Reset each get_move call; capped below the killer score band (8000-9000).
_history: list[list[int]] = [[0] * 64 for _ in range(64)]

# Countermove heuristic: best response to the previous move.
# Scored between killers (8000-9000) and history.
_counter: list[list[chess.Move | None]] = [[None] * 64 for _ in range(64)]


class _Timeout(Exception):
    pass


def _has_non_pawn_material(board: chess.Board) -> bool:
    color = board.turn
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if board.pieces(pt, color):
            return True
    return False


def _get_legal_moves(board: chess.Board) -> list[chess.Move]:
    bbs, stm = encode(board)
    ep_sq = board.ep_square if board.ep_square is not None else -1
    candidates = generate_pseudo_legal(bbs, stm, ep_sq, board.castling_rights)
    return [m for m in candidates if board.is_legal(m)]


def _get_captures(board: chess.Board) -> list[chess.Move]:
    """Legal captures and queen promotions — moves searched in quiescence."""
    bbs, stm = encode(board)
    ep_sq = board.ep_square if board.ep_square is not None else -1
    candidates = generate_pseudo_legal(bbs, stm, ep_sq, board.castling_rights)
    return [
        m for m in candidates
        if board.is_legal(m) and (board.is_capture(m) or m.promotion == chess.QUEEN)
    ]


def _move_score(
    board: chess.Board,
    move: chess.Move,
    tt_move: chess.Move | None,
    killers: list[chess.Move | None],
    prev_move: chess.Move | None = None,
) -> int:
    if move == tt_move:
        return 20_000
    victim = board.piece_type_at(move.to_square)
    if victim is not None or board.is_en_passant(move):
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        ep_victim = chess.PAWN if board.is_en_passant(move) else 0
        mvv = victim if victim is not None else ep_victim
        return 10_000 + mvv * 10 - attacker
    if killers[0] is not None and move == killers[0]:
        return 9_000
    if killers[1] is not None and move == killers[1]:
        return 8_000
    if prev_move is not None:
        c = _counter[prev_move.from_square][prev_move.to_square]
        if c is not None and move == c:
            return 7_500
    return _history[move.from_square][move.to_square]


def _lva_sq(to_sq: int, occ: int, is_white: bool, board: chess.Board) -> int:
    """Return the square of the least-valuable piece of `is_white` that attacks `to_sq`
    given occupancy `occ`. Returns -1 if no attacker exists."""
    occ_u = np.uint64(occ)
    color_co = chess.WHITE if is_white else chess.BLACK
    color_bb = int(board.occupied_co[color_co]) & occ
    # White pawns that attack to_sq sit on squares attacked by a black pawn FROM to_sq (idx=1).
    # Black pawns that attack to_sq sit on squares attacked by a white pawn FROM to_sq (idx=0).
    pawn_atk_idx = 1 if is_white else 0  # int, not bool — Numba requires int array index
    bb = int(pawn_attacks_bb(pawn_atk_idx, to_sq)) & color_bb & int(board.pawns)
    if bb:
        return (bb & -bb).bit_length() - 1
    bb = int(knight_attacks_bb(to_sq)) & color_bb & int(board.knights)
    if bb:
        return (bb & -bb).bit_length() - 1
    bb = int(bishop_attacks_bb(to_sq, occ_u)) & color_bb & int(board.bishops)
    if bb:
        return (bb & -bb).bit_length() - 1
    bb = int(rook_attacks_bb(to_sq, occ_u)) & color_bb & int(board.rooks)
    if bb:
        return (bb & -bb).bit_length() - 1
    bb = int(queen_attacks_bb(to_sq, occ_u)) & color_bb & int(board.queens)
    if bb:
        return (bb & -bb).bit_length() - 1
    bb = int(king_attacks_bb(to_sq)) & color_bb & int(board.kings)
    if bb:
        return (bb & -bb).bit_length() - 1
    return -1


def _see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation. Returns expected net material gain for the mover."""
    to_sq = move.to_square
    target_pt = board.piece_type_at(to_sq)
    if target_pt is None:
        if not board.is_en_passant(move):
            return 0
        target_val = _PIECE_VAL[chess.PAWN]
    else:
        target_val = _PIECE_VAL[target_pt]

    moving_pt = board.piece_type_at(move.from_square) or chess.PAWN
    if move.promotion is not None:
        target_val += _PIECE_VAL[move.promotion] - _PIECE_VAL[chess.PAWN]
        moving_pt = move.promotion

    occ = int(board.occupied) ^ (1 << move.from_square)
    if board.is_en_passant(move):
        ep_sq = move.to_square + (8 if board.turn == chess.BLACK else -8)
        occ ^= 1 << ep_sq

    gain: list[int] = [target_val]
    current_pt = moving_pt
    is_white = not board.turn  # opponent recaptures first

    while True:
        sq = _lva_sq(to_sq, occ, is_white, board)
        if sq < 0:
            break
        attacker_pt = board.piece_type_at(sq) or chess.PAWN
        gain.append(_PIECE_VAL[current_pt] - gain[-1])
        current_pt = attacker_pt
        occ ^= 1 << sq
        is_white = not is_white

    # Backward negamax: each player can choose not to recapture.
    for i in range(len(gain) - 2, 0, -1):
        gain[i - 1] = -max(-gain[i - 1], gain[i])

    return gain[0]


def quiesce(board: chess.Board, alpha: float, beta: float, deadline: float, ply: int = 0) -> float:
    """Capture search until quiet. Fixes the horizon effect."""
    if time.monotonic() >= deadline:
        raise _Timeout()

    bbs, stm = encode(board)
    stand_pat = float(evaluate(bbs, stm))

    in_check_now = board.is_check()
    if in_check_now:
        moves = _get_legal_moves(board)
        if not moves:
            return float(-(MATE - ply))
        moves.sort(key=lambda m: _move_score(board, m, None, _NO_KILLERS), reverse=True)
    else:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        # Position-level delta: if even winning a queen + buffer can't reach alpha, give up.
        if stand_pat + _MAX_GAIN < alpha:
            return alpha
        # SEE filter and sort: skip losing captures, order winning ones first.
        all_captures = _get_captures(board)
        see_scores = {id(m): _see(board, m) for m in all_captures}
        moves = [m for m in all_captures if see_scores[id(m)] >= 0]
        moves.sort(key=lambda m: see_scores[id(m)], reverse=True)

    for move in moves:
        # Per-capture delta: skip if gaining this piece for free can't reach alpha.
        if not in_check_now:
            victim = board.piece_type_at(move.to_square)
            victim_val = _PIECE_VAL[victim] if victim is not None else _PIECE_VAL[chess.PAWN]
            if stand_pat + victim_val + _DELTA_MARGIN < alpha:
                continue
        board.push(move)
        score = -quiesce(board, -beta, -alpha, deadline, ply + 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def alphabeta(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    deadline: float,
    ply: int = 0,
    excluded_move: chess.Move | None = None,
    prev_move: chess.Move | None = None,
) -> float:
    if time.monotonic() >= deadline:
        raise _Timeout()

    # Draw by repetition or 50-move rule.
    if board.is_repetition(2) or board.halfmove_clock >= 100:
        return 0.0

    key = cast(int, board._transposition_key())
    entry = _tt.probe(key)
    tt_move: chess.Move | None = entry.move if entry is not None else None

    # TT cutoff — skip when inside a singular probe to get a fresh result.
    if excluded_move is None and entry is not None and entry.depth >= depth:
        tt_score = _tt_denorm(entry.score, ply)
        if entry.flag == Flag.EXACT:
            return tt_score
        if entry.flag == Flag.LOWER_BOUND:
            alpha = max(alpha, tt_score)
        elif entry.flag == Flag.UPPER_BOUND:
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score

    moves = _get_legal_moves(board)
    if not moves:
        return float(-(MATE - ply)) if board.is_check() else 0.0

    if depth == 0:
        return quiesce(board, alpha, beta, deadline, ply)

    in_check = board.is_check()

    # Null-move pruning — disabled in check and in king+pawn endings (zugzwang risk).
    if (
        excluded_move is None
        and depth >= _NMP_R + 1
        and not in_check
        and _has_non_pawn_material(board)
    ):
        board.push(chess.Move.null())
        null_score = -alphabeta(board, depth - 1 - _NMP_R, -beta, -beta + 1.0, deadline, ply + 1)
        board.pop()
        if null_score >= beta:
            return beta

    # IID — shallow search to get a TT move when none is cached.
    if excluded_move is None and tt_move is None and depth >= _IID_MIN_DEPTH:
        alphabeta(board, depth - 2, alpha, beta, deadline, ply)
        iid_entry = _tt.probe(key)
        if iid_entry is not None:
            tt_move = iid_entry.move

    # Singular extension probe.
    singular_move: chess.Move | None = None
    if (
        excluded_move is None
        and depth >= _SE_MIN_DEPTH
        and not in_check
        and tt_move is not None
        and entry is not None
        and entry.depth >= depth - 3
        and entry.flag != Flag.UPPER_BOUND
        and abs(entry.score) < _MATE_THRESHOLD
    ):
        s_beta = _tt_denorm(entry.score, ply) - _SE_MARGIN
        s_score = alphabeta(
            board, max(1, depth // 2), s_beta - 1.0, s_beta,
            deadline, ply, excluded_move=tt_move,
        )
        if s_score < s_beta:
            singular_move = tt_move

    # Futility pruning at depth 1: if the static eval plus a one-pawn buffer can't
    # reach alpha, quiet non-special moves are unlikely to help.
    futile = False
    if (
        excluded_move is None
        and depth == 1
        and not in_check
        and abs(alpha) < _MATE_THRESHOLD
        and abs(beta) < _MATE_THRESHOLD
    ):
        bbs, stm = encode(board)
        static_eval = float(evaluate(bbs, stm))
        if static_eval + _FUTILITY_MARGIN <= alpha:
            futile = True

    killers = _killers[ply] if ply < len(_killers) else _NO_KILLERS
    moves.sort(key=lambda m: _move_score(board, m, tt_move, killers, prev_move), reverse=True)

    orig_alpha = alpha
    best_score = -math.inf
    best_move_at_node: chess.Move | None = None

    for i, move in enumerate(moves):
        if excluded_move is not None and move == excluded_move:
            continue

        is_quiet = (
            board.piece_type_at(move.to_square) is None
            and not board.is_en_passant(move)
            and move.promotion is None
        )
        is_killer = (killers[0] is not None and move == killers[0]) or (
            killers[1] is not None and move == killers[1]
        )

        # Skip quiet non-killer non-TT moves when futility is triggered.
        if futile and is_quiet and not is_killer and move != tt_move:
            continue

        board.push(move)
        gives_check = board.is_check()

        # Extensions: singular or check, capped at 1.
        ext = 1 if (singular_move is not None and move == tt_move) or gives_check else 0

        # LMR: log-based reduction for quiet non-special moves searched late.
        lmr_reduction = 0
        if (
            ext == 0
            and depth >= _LMR_MIN_DEPTH
            and i >= _LMR_MIN_MOVE_INDEX
            and is_quiet
            and not in_check
            and not gives_check
            and move != tt_move
            and not is_killer
        ):
            lmr_reduction = max(1, int(math.log(depth) * math.log(i + 1) / 2))

        if lmr_reduction > 0:
            score = -alphabeta(
                board, depth - 1 - lmr_reduction, -alpha - 1.0, -alpha, deadline, ply + 1,
                prev_move=move,
            )
            if score > alpha:
                score = -alphabeta(
                    board, depth - 1, -beta, -alpha, deadline, ply + 1, prev_move=move,
                )
        else:
            score = -alphabeta(
                board, depth - 1 + ext, -beta, -alpha, deadline, ply + 1, prev_move=move,
            )

        board.pop()

        if score >= beta:
            if is_quiet:
                if killers[0] is None or move != killers[0]:
                    killers[1] = killers[0]
                    killers[0] = move
                h = _history[move.from_square][move.to_square] + depth * depth
                _history[move.from_square][move.to_square] = h if h < 7_000 else 7_000
                if prev_move is not None:
                    _counter[prev_move.from_square][prev_move.to_square] = move
            if excluded_move is None:
                _tt.store(key, TTEntry(depth, _tt_norm(score, ply), Flag.LOWER_BOUND, move))
            return beta

        if score > best_score:
            best_score = score
            best_move_at_node = move
        if score > alpha:
            alpha = score

    # Guard: don't store -inf (all moves were futility-pruned with no captures).
    if excluded_move is None and best_score > -math.inf:
        flag = Flag.UPPER_BOUND if best_score <= orig_alpha else Flag.EXACT
        _tt.store(key, TTEntry(depth, _tt_norm(best_score, ply), flag, best_move_at_node))
    return alpha


def _root_search(
    board: chess.Board,
    moves: list[chess.Move],
    depth: int,
    alpha: float,
    beta: float,
    deadline: float,
) -> tuple[float, chess.Move | None]:
    """One pass over root moves within [alpha, beta]. Raises _Timeout."""
    best_score = alpha
    candidate: chess.Move | None = None
    for move in moves:
        board.push(move)
        score = -alphabeta(board, depth - 1, -beta, -best_score, deadline, ply=1, prev_move=move)
        board.pop()
        if score > best_score:
            best_score = score
            candidate = move
        if score >= beta:
            break
    return best_score, candidate


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    tm = TimeManager(time_left_ms, board)

    for k in _killers:
        k[0] = k[1] = None
    for row in _history:
        row[:] = [0] * 64
    for crow in _counter:
        crow[:] = [None] * 64

    moves = _get_legal_moves(board)
    assert moves
    best_move = moves[0]
    prev_score: float = 0.0
    root_key = cast(int, board._transposition_key())
    time_extended = False

    for depth in range(1, _MAX_DEPTH + 1):
        if depth > 2:
            search_windows: list[tuple[float, float]] = [
                (prev_score - _ASPIRATION_DELTA, prev_score + _ASPIRATION_DELTA),
                (-MATE, MATE),
            ]
        else:
            search_windows = [(-MATE, MATE)]

        timed_out = False
        score = prev_score
        candidate: chess.Move | None = None

        for lo, hi in search_windows:
            root_entry = _tt.probe(root_key)
            root_tt_move = root_entry.move if root_entry is not None else None
            moves.sort(
                key=lambda m: _move_score(board, m, root_tt_move, _NO_KILLERS),
                reverse=True,
            )

            try:
                score, candidate = _root_search(board, moves, depth, lo, hi, tm.hard_deadline)
            except _Timeout:
                timed_out = True
                break
            if lo < score < hi:
                break

        if timed_out:
            break
        if candidate is not None:
            # Extend soft deadline when the score swings sharply between depths.
            if (
                not time_extended
                and depth > 1
                and abs(score - prev_score) >= _INSTABILITY_THRESHOLD
            ):
                tm.extend(_INSTABILITY_FACTOR)
                time_extended = True
            best_move = candidate
            prev_score = score
            _tt.store(root_key, TTEntry(depth, _tt_norm(score, 0), Flag.EXACT, best_move))
        if tm.should_stop():
            break

    return best_move.uci()
