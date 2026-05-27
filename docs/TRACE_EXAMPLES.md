# Trace Examples (Exported From Traces)

This document is generated directly from proof traces under `tests/logs/`.

| id | category | input | expected | observed | verdict | root_cause | argument_vector | source_reliability | failure_severity | reproducibility_status | trace_ref | notes |
|---:|---|---|---|---|---|---|---|---:|---|---|---|---|
| 1 | Cyclic inheritance propagation | `A is B; B is C; C is A; C can fly; Query: Do A fly?` | N/A | YES | INFO | taxonomic_inheritance | +can_fly[s=2,r=1,rel=1.0,a=0.5] | 1.0 | N/A | deterministic | tests/logs/proof_do_a_fly_1779187650.json | trace-backed |
| 2 | Penguin exception | `Birds can fly; Penguins are birds; Penguins cannot fly; Query: Do penguins fly?` | N/A | NO | INFO | hard_negative | - | - | N/A | deterministic | tests/logs/proof_do_penguins_fly_1779187650.json | - |
| 3 | Nixon diamond | `Quakers are pacifists; Republicans are not pacifists; Nixon is a quaker; Nixon is a republican; Query: Is Nixon a pacifist?` | N/A | CONFLICT | INFO | conflict_resolution | +nixon is pacifist[s=2,r=1,rel=1.0,a=0.9] ; !nixon is pacifist[s=2,r=2,rel=1.0,a=0.9] ; conflict[pos=?,neg=?] | 1.0 | N/A | deterministic | tests/logs/proof_is_nixon_a_pacifist_1779187650.json | trace-backed |

## Why These Rows Matter

- These are positive validation artifacts, not failures.
- They are useful for explainability, replay, and regression comparisons.
