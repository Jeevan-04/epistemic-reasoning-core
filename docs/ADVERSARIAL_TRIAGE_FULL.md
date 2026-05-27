# Adversarial Triage Summary



Quick counts by root cause:

- **unknown**: 35

- **missing_proof**: 15



# Adversarial Triage (full)

## misc

- **mi-001**: Penguin inheritance basic — expected: **NO**, observed: **no**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-001.json))
- **mi-002**: Military penguins override — expected: **YES**, observed: **yes**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-002.json))
- **mi-003**: Conflicting subclass rules — expected: **YES**, observed: **yes**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-003.json))
- **mi-004**: Contradiction from two parents — expected: **CONFLICT**, observed: **conflict**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-004.json))
- **mi-005**: Priority by rank — expected: **NO**, observed: **no**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-005.json))
- **mi-006**: Inherited exception chain — expected: **UNKNOWN**, observed: **yes**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_mi-006.json))
- **mi-007**: Diamond inheritance with opposing defaults — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-007.json))
- **mi-008**: Diamond with conflicting intermediates — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-008.json))
- **mi-009**: Specificity wins over strength — expected: **swimming**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-009.json))
- **mi-010**: Conflicting deep inheritance — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mi-010.json))
- **rel-001**: Trusted vs untrusted source — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-001.json))
- **rel-002**: Many weak sources vs one strong — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-002.json))
- **rel-003**: Source reliability tie — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-003.json))
- **rel-004**: Aggregate weak support — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_rel-004.json))
- **rel-005**: Temporal reliability shift — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-005.json))
- **rel-006**: Chain of trust — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-006.json))
- **rel-007**: Source with partial reliability — expected: **CONFLICT**, observed: **yes**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-007.json))
- **rel-008**: Misinformation cascade — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_rel-008.json))
- **act-001**: Old strong belief vs recent weak override — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_act-001.json))
- **act-002**: Recency beats old strong — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_act-002.json))
- **act-003**: Decay leading to abstention — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_act-003.json))
- **act-004**: Gradual override — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_act-004.json))
- **act-005**: Activation threshold test — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_act-005.json))
- **act-006**: Temporal aggregation — expected: **NO**, observed: **yes**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_act-006.json))
- **act-007**: Short-lived spike — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_act-007.json))
- **act-008**: Decay plus reinforcement — expected: **YES**, observed: **yes**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_act-008.json))
- **cyc-001**: Simple cycle — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_cyc-001.json))
- **cyc-002**: Cycle with exception — expected: **CONFLICT**, observed: **conflict**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_cyc-002.json))
- **cyc-003**: Cycle with specificity — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_cyc-003.json))
- **cyc-004**: Mutually recursive defaults — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_cyc-004.json))
- **cyc-005**: Large taxonomy cycle — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_cyc-005.json))
- **cyc-006**: Exception that closes the cycle — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_cyc-006.json))
- **abs-001**: No evidence — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_abs-001.json))
- **abs-002**: Balanced weak evidence — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_abs-002.json))
- **abs-003**: Soft support without commitment — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_abs-003.json))
- **abs-004**: Conflicting strong supports — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_abs-004.json))
- **abs-005**: Abstention vs explicit no — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_abs-005.json))
- **abs-006**: Partial abstention aggregation — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_abs-006.json))
- **abs-007**: Abstain vs low negative — expected: **NO**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_abs-007.json))
- **abs-008**: Ambiguity propagation — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_abs-008.json))
- **exp-001**: Combinatorial rule expansion — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_exp-001.json))
- **exp-002**: Redundant derivations — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_exp-002.json))
- **exp-003**: Cross-coupled rules — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_exp-003.json))
- **exp-004**: Large conjunctions — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_exp-004.json))
- **exp-005**: Filtered majority — expected: **YES**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_exp-005.json))
- **mix-001**: Reliability + inheritance clash — expected: **NO**, observed: **no**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mix-001.json))
- **mix-002**: Activation, reliability, and abstention — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mix-002.json))
- **mix-003**: Deep inheritance + cycle — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_mix-003.json))
- **mix-004**: Adversarial default ordering — expected: **CONFLICT**, observed: **unknown**, replay: **unknown** — root cause: *unknown* ([proof](tests/logs/proofs/proof_mix-004.json))
- **mix-005**: Multiple-source temporal war — expected: **UNKNOWN**, observed: **unknown**, replay: **unknown** — root cause: *missing_proof* ([proof](tests/logs/proofs/proof_mix-005.json))

