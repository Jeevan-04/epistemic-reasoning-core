Reproducibility notes
====================

To reproduce the evaluation artifacts and figures used in the paper:

1. Install Python dependencies for figure generation:

```bash
python3 -m pip install -r requirements.txt
```

2. Run the benchmark runners (this may take a while):

```bash
PYTHONPATH=. python3 scripts/run_parser_benchmarks.py --bench tests/benchmarks/adversarial_benchmarks_nl.json
PYTHONPATH=. python3 scripts/run_symbolic_reasoning_benchmarks.py
```

3. Aggregate results and generate tables/figures:

```bash
PYTHONPATH=. python3 scripts/aggregate_benchmarks.py
PYTHONPATH=. python3 scripts/generate_paper_eval_artifacts.py
PYTHONPATH=. python3 scripts/generate_figures.py
```

4. Optional: bundle results for sharing:

```bash
bash scripts/create_repro_bundle.sh
```

5. To compile the paper locally you will need a LaTeX installation (e.g., TeX Live):

```bash
pdflatex paper/episteme_paper.tex
```

If you are missing any dependency (e.g., `matplotlib`), the figure generator will
print installation instructions.
# Reproducibility Instructions

Quickstart (minimal steps to reproduce tests and benchmarks):

1. Clone the repository

```bash
git clone https://your-repo-url Episteme
cd Episteme
```

2. Create a Python virtual environment and install requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the unit tests

```bash
pytest -q
```

4. Run the mini benchmark

```bash
python bench/compare_baselines.py
```

5. Replay a saved trace

```bash
python3 replay_trace.py tests/logs/proof_is_nixon_a_pacifist_1779187650.json
```

This rebuilds the reasoning pipeline from the trace's teachings, reruns the
query, and compares the replayed verdict against the stored one.

Notes

- If you do not have Python 3.10+, adjust the venv command accordingly.
- The `requirements.txt` is intentionally minimal; add packages as needed for
  any optional baselines (e.g., Prolog/PySWIP).

How we ensure reproducibility

- All unit tests are deterministic and small.
- Proof traces are saved to `tests/logs/` when tests or benchmarks are extended
  to capture trace examples (see `docs/TRACE_EXAMPLES.md`) and confirmed failures
  (see `docs/FAILURE_TABLES.md`).
