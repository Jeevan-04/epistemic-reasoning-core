# Research Journal — Episteme Adversarial Benchmark

Date: 2026-05-27

## 2026-05-27 — Documentation reset and repo cleanup

Summary
- Rewrote the README into a grounded research overview with architecture and build diagrams.
- Aligned the documentation surface with the formal spec and current research-status notes.
- Marked generated artifacts as transient so the repository stays focused on source and paper inputs.

Files changed
- README.md
- .gitignore
- paper/episteme_paper.tex

Hypothesis
- A shorter, architecture-first README plus a stricter generated-artifact policy will make the repository easier to review, reproduce, and publish without inflating claims.

Protocol
- Reviewed the current manuscript, formal docs, and repository layout.
- Replaced legacy README content with a concise research summary and diagrams.
- Added ignore rules for generated outputs and benchmark logs.

Outcome
- The repository documentation now reflects the current modular architecture and the paper workflow more directly.
- Generated benchmark and figure outputs are treated as disposable build artifacts rather than source.

Next
- Finish PDF compilation once the LaTeX toolchain is available.
- Push the cleaned, documentation-focused revision after validation.

Objective
- Triage adversarial benchmark failures and prioritize algorithmic fixes for robust default reasoning.

Summary of recent work
- 2026-05-22: Ran full adversarial benchmark (50 cases) via `scripts/run_adversarial_benchmarks.py`.
- Exported results to `tests/logs/adversarial_results.jsonl` and per-case proofs into `tests/logs/proofs/`.
- Initial automated summary: all 50 cases returned `unknown` (no decisive derivation); proofs available for triage.
- 2026-05-22: Added a controlled symbolic reasoning benchmark at `tests/benchmarks/reasoning_symbolic_benchmarks.json` and runner `scripts/run_symbolic_reasoning_benchmarks.py`.
- Validated the split pipeline on a 5-case slice: controlled symbolic cases now produce meaningful verdicts and proof traces without Manas parsing.

Key observations
- Automated run (50 cases) produced 50 `unknown` verdicts. This indicates a systematic grounding or applicability gating issue in the current pipeline.
- Early triage points to:
   - `predicate grounding` failures: predicates in queries (e.g., `can_fly` vs `fly`) are not matching taught beliefs in many cases.
   - `entity normalization` / parser canonicalization: placeholder tokens and pluralization create lookup misses (e.g., `X`, `Bat-owl` variants).
   - Strict primary-entity existence gate in `buddhi.answer()` causes immediate `unknown` when the parser-normalized entity doesn't appear in `chitta.entity_index`.

Methodological takeaway
- The benchmark is currently conflating parsing and reasoning. The natural-language cases are useful for end-to-end stress testing, but they are too noisy to isolate pure defeat ordering or inheritance quality.
- The next research step should separate evaluation into:
   - a controlled symbolic benchmark for reasoning semantics, and
   - a parser-integrated benchmark for canonicalization and grounding.
- This will make failures localizable and turn the current `unknown` wall into stage-specific metrics such as parse success, grounding success, argument construction rate, and final verdict resolution.

Next experiments

1. Defeat-ordering patch: implement and test alternate tiebreakers in `buddhi._resolve_taxonomic_conflict()`.
   - Hypothesis: prioritizing specificity then source_reliability then activation will resolve many `multi_inheritance` cases.
   - Test: add targeted unit tests for `mi-001..mi-010` and assert expected verdicts.

2. Grounding improvements: expand `manas.parse()` normalization and `chitta` indexing to handle placeholders, plurals, and shorthand notation.
   - Hypothesis: removing grounding errors will reduce `entity_grounding` labels and improve recall.

3. Argument export consistency: ensure `AnswerProof.metadata.arguments` includes canonical fields for defeat-order analysis.
   - Save enriched proofs for later statistical analysis.


How to reproduce current triage

```bash
PYTHONPATH=. python3 scripts/run_adversarial_benchmarks.py --bench tests/benchmarks/adversarial_benchmarks_natural.json --limit 50
PYTHONPATH=. python3 scripts/triage_mismatches.py
python3 replay_trace.py tests/logs/proofs/proof_mi-001.json

Quick stats (2026-05-22 run):

- Total cases: 50
- Verdict distribution: `unknown`: 50
- Mismatches vs expected: 0 (automated run produced `unknown` for all cases)

Files produced:

- `tests/logs/adversarial_results.jsonl` (50 JSONL rows)
- `tests/logs/proofs/` (per-case proof_*.json traces)
```

Notes & open questions
- Do we prefer specificity-over-reliability, or a weighted combination? (needs policy decision)
- Should `manas` aggressively canonicalize placeholders (X→entity) during parsing, or should the runner handle substitution? Prefer parser-side canonicalization for consistency.

Next steps (operational)
- Finish the parser benchmark half of the split by adding a natural-language -> symbolic translation harness.
- Extend the symbolic benchmark with reliability and activation cases once the parser track is separated.
- Keep the natural-language adversarial suite as an end-to-end parser-integrated stress test, not the primary reasoning benchmark.

I can start triaging the exported proofs now and generate a concise failure table. Proceed? (reply 'proceed' to start)
# Research Journal

This file records technical evolution notes, hypotheses, failures, and benchmark impacts.

## Template
- Date: YYYY-MM-DD
- Title: short descriptive title
- Summary: 1-2 line summary of the change/experiment
- Files changed: list of file paths
- Hypothesis: what we expected to happen
- Protocol: commands or steps used to run the experiment
- Outcome: short result summary and links to traces/logs
- Next: follow-up actions

---

## 2026-05-20 — Seed: adversarial benchmarks added
- Summary: Added 50 adversarial benchmark cases across categories (multi-inheritance, reliability conflicts, activation/decay, cyclic taxonomy, abstention, argument explosion, mixed complexity).
- Files changed: `tests/benchmarks/adversarial_benchmarks.json`, `scripts/run_adversarial_benchmarks.py`
- Hypothesis: These cases will expose weaknesses in defeat ordering, activation decay handling, and ambiguity propagation.
- Protocol:
  - `PYTHONPATH=. python3 scripts/run_adversarial_benchmarks.py --limit 50`
  - Inspect `tests/logs/adversarial_results.jsonl` and per-case traces `tests/logs/bench_*.json`
- Outcome: (to be filled after running benchmarks)
- Next: run full batch, triage failures into `docs/FAILURE_TABLES.md` and add reproducible traces to `docs/TRACE_EXAMPLES.md`.
