SHELL := /bin/bash

.PHONY: setup play arena zip gate test train

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
