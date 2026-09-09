SHELL := /bin/bash

.PHONY: setup play arena zip gate test train selfplay train-gpu-setup train-gpu

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/v3 --games 20

zip:
	uv run python -m harness.package --include src --include weights

test:
	uv run python -m pytest -q

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m pytest -q
	uv run python -m harness.arena --opponent baselines/v3 --games 2 --base-ms 5000

train:
	uv run python -m train.generate_data --pgn $(PGN) --stockfish $(SF) --out train/data.npz
	uv run python -m train.train_nnue --data train/data.npz --out train/nnue.pt
	uv run python -m train.export_weights --checkpoint train/nnue.pt --out weights/nnue.npz

selfplay:
	uv run python -m train.selfplay --games-per-pair $(if $(GPP),$(GPP),20) --out train/selfplay.pgn

train-gpu-setup:
	uv venv train/.venv-gpu --python 3.12
	uv pip install --python train/.venv-gpu/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
	uv pip install --python train/.venv-gpu/bin/python numpy==2.5.2 chess==1.11.2

train-gpu:
	uv run python -m train.generate_data --pgn train/selfplay.pgn --stockfish /usr/games/stockfish \
	    --samples-per-game 16 --limit 200000 --out train/data.npz
	train/.venv-gpu/bin/python -m train.train_nnue --data train/data.npz --out train/nnue.pt --workers 6
	uv run python -m train.export_weights --checkpoint train/nnue.pt --out weights/nnue.npz
