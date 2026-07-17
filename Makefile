.PHONY: help test lint bench report clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

test:  ## Run the deterministic suite (stub model — no API key, no Ollama needed)
	pytest -q

lint:  ## Ruff lint
	ruff check src tests

bench:  ## Run the config matrix -> results (needs a model backend: [local] or [api])
	python -m memprobe.harness.run --config config/default.yaml

report:  ## Render the results-first markdown + Pareto plot from the latest run
	python -m memprobe.report.render --run data/runs/latest

clean:  ## Remove generated run artifacts
	rm -rf data/runs report/out .pytest_cache .ruff_cache
