.PHONY: install test demo plan lint
install:
	python -m pip install -e .[dev]
test:
	pytest -q
demo:
	g2mrf demo --config configs/default.yaml --out results/demo_report.json
plan:
	g2mrf plan --h2 0.35 --me 75000 --target 0.05
lint:
	ruff check src tests
