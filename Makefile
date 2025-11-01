SHELL := /bin/bash
PYTHON := python3

.PHONY: all setup lint sim cov runtime test docs bench clean

all: runtime test

setup:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r scripts/requirements.txt

lint:
	verilator --lint-only rtl/top/gb10_soc_top.sv rtl/cpu/gb10_cpu_cluster.sv rtl/gpu/gb10_gpu_core.sv rtl/interconnect/axi_noc_router.sv rtl/memory/lpddr5x_controller.sv
	$(PYTHON) -m scripts.lint

sim:
	$(PYTHON) scripts/run_sim.py

cov:
	$(PYTHON) scripts/run_cov.py

runtime:
	cmake -S runtime/libquasim -B build/libquasim -DCMAKE_BUILD_TYPE=RelWithDebInfo
	cmake --build build/libquasim

bench: runtime
	PYTHONPATH=runtime/python:quantum $(PYTHON) benchmarks/quasim_bench.py

install:
	cmake --install build/libquasim --prefix install

test:
	PYTHONPATH=runtime/python:quantum $(PYTHON) -m pytest -q

docs:
	$(PYTHON) -m scripts.build_docs

clean:
	rm -rf build install .venv
	rm -f tests/software/*.pyc runtime/python/quasim/__pycache__
