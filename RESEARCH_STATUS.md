# Episteme Research Status

This file is the truth reset for the current repository state. It records what
has been observed in the current checkout and deliberately avoids aspirational
claims.

## Current Research Position

Episteme is an experimental Python prototype for proof-carrying symbolic memory.
It currently explores a separation between:

- symbolic verdicts: `YES`, `NO`, `UNKNOWN`, `CONFLICT`, `INVALID`
- belief storage and retrieval availability
- direct assertion, taxonomic inference, explicit negation, and abstention

The project is not yet a publishable research system. It still lacks complete
formal semantics, complexity analysis, baseline comparisons, ablations, and a
fully finalized experimental section for publication.

## Research Boundary

Episteme is being scoped as a systems paper about:

- proof-carrying symbolic reasoning
- explicit argument traces
- defeasible conflict handling
- parser/reasoner separation
- adversarial reasoning evaluation

It is **not** being framed as:

- AGI
- cognition
- universal reasoning
- natural-language understanding
- a complete theory of intelligence

This boundary is deliberate and should be kept consistent in the README, paper
draft, benchmark names, and future status updates.

## Baseline v0

The current codebase should be treated as `baseline-v0`: a runnable prototype
whose behavior is measured before the formal rebuild.

Observed before the revival work:

- `python -m unittest discover tests`: 9 tests, 3 failures.
- `python tests/run_scientific_benchmark.py`: 10/16 passed, 62.5%.
- `python tests/benchmark_strict.py`: 516/1050 passed, 49.1%.
- `python showcase_episteme.py`: ran, but the specificity and Nixon conflict
  demonstrations returned `UNKNOWN` instead of the expected `NO` and `CONFLICT`.

These numbers supersede older README and paper claims until new reproducible
results are generated.

Observed after the first revival pass:

- `python -m unittest discover tests`: 9 tests, all passing.
- `python tests/run_scientific_benchmark.py`: 14/16 passed, 87.5%.
- `python tests/benchmark_strict.py`: 517/1050 passed, 49.2%.
- `python tests/run_research_benchmark.py`: seed family benchmark created;
  `episteme_full` passed 8/10. It still fails the current Nixon-style
  horizontal conflict and source-reliability conflict seed cases.
- `python scripts/reproduce_current_results.py`: runs unit tests, scientific
  benchmark, strict benchmark, and showcase; writes
  `tests/logs/reproducibility_summary.json`.

## Working Capabilities

- Stores parsed beliefs in `ChittaGraph`.
- Persists graph state to JSON.
- Handles some direct facts and some taxonomic membership chains.
- Handles some explicit negative capability queries, especially `Can X Y?`.
- Provides proof-step traces through `Buddhi.AnswerProof`.
- Supports a clone-based hypothetical sandbox through HRE.
- Implements an event-observer utility in Sakshin.
- Supports two benchmark modes:
  - controlled symbolic reasoning (reasoning-only evaluation)
  - natural-language integrated parsing/grounding/reasoning evaluation

## Known Failure Modes

- Parser normalization has historically been inconsistent across subjects,
  objects, entities, and query forms.
- Semantically similar questions can produce different predicates.
- Some taxonomic inference paths are incomplete.
- Some class-membership queries can be over-accepted if property inheritance is
  allowed to answer `is_a` questions.
- The strict benchmark runner previously printed poor accuracy but exited with
  success.
- Documentation and paper drafts contained stale benchmark claims.

## Near-Term Research Goal

The immediate target is not to claim broad superiority over Prolog, ASP, Cyc,
TMS, MLNs, PSL, knowledge graph reasoners, or cognitive architectures.

The defensible target is narrower:

> Dynamic proof-carrying epistemic memory under open-world abstention,
> defeasible conflict, separated truth/activation semantics, and a measurable
> split between parser quality and reasoning quality.

## Publication Framing

The paper should emphasize:

1. controlled symbolic reasoning evaluation,
2. natural-language degradation analysis,
3. stage-level metrics that isolate parser, grounding, argument construction,
   and verdict resolution failures.
