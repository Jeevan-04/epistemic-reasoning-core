# Adversarial Failures — Summary

Generated from `tests/logs/adversarial_results.jsonl` and detailed triage at `docs/ADVERSARIAL_TRIAGE_FULL.md`.

Summary (run date: 2026-05-21):

- Total benchmark cases: 50
- Mismatches: 35

Top categories (mismatch counts):

- multi_inheritance: 9
- reliability_conflict: 7
- activation_decay: 5
- cyclic_taxonomy: 4
- abstention_edge: 4
- argument_explosion: 3
- mixed_complexity: 3

Observed root-cause labels (from triage):

- insufficient_inference — 27 cases (reasoner did not generate decisive derivation)
- entity_grounding — 6 cases (parser/entity mismatch or missing grounding)
- no_derivation / other — 2 cases

Representative examples and suggested fixes

- `mi-001` (Penguin inheritance basic): reasoner fails to apply exception specificity chain. Suggested fix: review `buddhi._apply_taxonomic_inference()` and defeat-ordering heuristics to prefer more specific subclass defaults.
- `mi-004` (Contradiction from two parents): triage shows grounding path; ensure `manas` produces concrete entity tokens and `chitta` indexing matches singular/plural forms.
- `rel-007` (Source with partial reliability): replay matched observed `yes` — investigate why original runner produced different observed output; ensure proof serialization (`AnswerProof.metadata.arguments`) is saved consistently.

Immediate action items (prioritized):

1. Implement targeted defeat-ordering improvements in `buddhi` for multi-inheritance and default conflicts. (high)
2. Strengthen entity grounding and normalization in `manas.parse()` and `chitta` indexing. (high)
3. Add unit tests for the 10 highest-impact mismatches under `tests/` and CI. (medium)
4. Improve argument-summary export: ensure `AnswerProof.metadata.arguments` contains `claim`, `is_negative`, `rank`, `specificity`, `source_reliability`, and `path`. (medium)
5. Add a deterministic replay CI job that runs `scripts/run_adversarial_benchmarks.py` and fails on unexpected mismatches. (low)

For full per-case triage, see: [docs/ADVERSARIAL_TRIAGE_FULL.md](docs/ADVERSARIAL_TRIAGE_FULL.md).
Per-case proofs are in `tests/logs/proofs/` (e.g., [tests/logs/proofs/proof_mi-001.json](tests/logs/proofs/proof_mi-001.json)).
# Adversarial Failures

This table lists benchmark cases where the observed verdict diverged from the expected semantics.

| id | title | category | expected | observed | trace | notes |
|---|---|---|---|---|---|---|
| mi-001 | Penguin inheritance basic | multi_inheritance | NO | unknown |  |  |
| mi-002 | Military penguins override | multi_inheritance | YES | unknown |  |  |
| mi-003 | Conflicting subclass rules | multi_inheritance | YES | unknown |  |  |
| mi-004 | Contradiction from two parents | multi_inheritance | CONFLICT | unknown |  |  |
| mi-005 | Priority by rank | multi_inheritance | NO | unknown |  |  |
| mi-007 | Diamond inheritance with opposing defaults | multi_inheritance | NO | unknown |  |  |
| mi-008 | Diamond with conflicting intermediates | multi_inheritance | CONFLICT | unknown |  |  |
| mi-009 | Specificity wins over strength | multi_inheritance | swimming | unknown |  |  |
| mi-010 | Conflicting deep inheritance | multi_inheritance | CONFLICT | unknown |  |  |
| rel-001 | Trusted vs untrusted source | reliability_conflict | YES | unknown |  |  |
| rel-002 | Many weak sources vs one strong | reliability_conflict | NO | unknown |  |  |
| rel-003 | Source reliability tie | reliability_conflict | CONFLICT | unknown |  |  |
| rel-005 | Temporal reliability shift | reliability_conflict | CONFLICT | unknown |  |  |
| rel-006 | Chain of trust | reliability_conflict | YES | unknown |  |  |
| rel-007 | Source with partial reliability | reliability_conflict | CONFLICT | yes |  |  |
| rel-008 | Misinformation cascade | reliability_conflict | NO | unknown |  |  |
| act-001 | Old strong belief vs recent weak override | activation_decay | YES | unknown |  |  |
| act-002 | Recency beats old strong | activation_decay | CONFLICT | unknown |  |  |
| act-004 | Gradual override | activation_decay | NO | unknown |  |  |
| act-006 | Temporal aggregation | activation_decay | NO | yes |  |  |
| act-007 | Short-lived spike | activation_decay | CONFLICT | unknown |  |  |
| cyc-001 | Simple cycle | cyclic_taxonomy | YES | unknown |  |  |
| cyc-002 | Cycle with exception | cyclic_taxonomy | CONFLICT | unknown |  |  |
| cyc-003 | Cycle with specificity | cyclic_taxonomy | NO | unknown |  |  |
| cyc-004 | Mutually recursive defaults | cyclic_taxonomy | CONFLICT | unknown |  |  |
| abs-002 | Balanced weak evidence | abstention_edge | CONFLICT | unknown |  |  |
| abs-004 | Conflicting strong supports | abstention_edge | CONFLICT | unknown |  |  |
| abs-005 | Abstention vs explicit no | abstention_edge | NO | unknown |  |  |
| abs-007 | Abstain vs low negative | abstention_edge | NO | unknown |  |  |
| exp-002 | Redundant derivations | argument_explosion | YES | unknown |  |  |
| exp-003 | Cross-coupled rules | argument_explosion | CONFLICT | unknown |  |  |
| exp-005 | Filtered majority | argument_explosion | YES | unknown |  |  |
| mix-001 | Reliability + inheritance clash | mixed_complexity | NO | unknown |  |  |
| mix-002 | Activation, reliability, and abstention | mixed_complexity | CONFLICT | unknown |  |  |
| mix-004 | Adversarial default ordering | mixed_complexity | CONFLICT | unknown |  |  |
