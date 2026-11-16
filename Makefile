# Short names for the commands used most.  `make help` lists them.
.PHONY: help install test lint format bench offline serve doctor dev test-run submit

PY ?= python3
CONFIG ?= configs/default.yaml
RUN ?= runs/default

help:            ## list the commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:         ## install Cairn with test and lint tools
	$(PY) -m pip install -e ".[dev]"

test:            ## run the test suite
	$(PY) -m pytest

lint:            ## check style and common mistakes
	ruff check cairn tests && ruff format --check cairn tests

format:          ## fix style automatically
	ruff check --fix cairn tests && ruff format cairn tests

bench:           ## measure everything that needs no model -> docs/results/offline_benchmarks.md
	cairn bench

offline:         ## full pipeline on 20 dev inputs per task with the stand-in model (no GPU)
	cairn --config configs/offline.yaml run --task all --split dev --limit 20 --run-dir runs/offline

serve:           ## start vLLM with the settings Cairn expects (needs a GPU)
	bash configs/serve/vllm.sh

doctor:          ## check what the running server supports
	cairn --config $(CONFIG) doctor

dev:             ## label the dev split with the real model and score it
	cairn --config $(CONFIG) run --task all --split dev --run-dir $(RUN)

test-run:        ## label the hidden-label test split
	cairn --config $(CONFIG) run --task all --split test --run-dir $(RUN)

submit:          ## validate and zip the eight prediction files
	cairn submit --run-dir $(RUN) --out submission.zip
