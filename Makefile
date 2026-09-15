.PHONY: install test eval benchmark export app clean gifs

## ── Setup ──────────────────────────────────────────────────────────────────
install:
	conda env create -f environment.yml

## ── Testing ─────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

test-policy:
	python -m pytest tests/test_policy.py -v

test-env:
	python -m pytest tests/test_environment.py -v

test-reward:
	python -m pytest tests/test_reward.py -v

## ── Intel Edge Pipeline ─────────────────────────────────────────────────────
export:
	python inference/export_openvino.py

benchmark:
	python inference/benchmark_intel.py

profile:
	python inference/latency_profile.py

## ── Training ─────────────────────────────────────────────────────────────────
train:
	python scripts/train_imitation.py --epochs 10

dataset:
	python scripts/generate_dataset.py --episodes 20 --steps 100

## ── Evaluation & Demo ────────────────────────────────────────────────────────
eval:
	python scripts/evaluate.py --seeds 10

stitch:
	python scripts/stitch_demo.py

gifs:
	python scripts/generate_gifs.py

## ── Web UI ───────────────────────────────────────────────────────────────────
app:
	python app.py

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t intel-vla-challenge .

docker-run:
	docker run -p 7860:7860 intel-vla-challenge

## ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
