# Adversarial Triage (sample 10 mismatches)

## mi-001 - Penguin inheritance basic
- expected: NO
- observed: unknown
- proof steps:
  - 1: focus -> Found 2 relevant belief(s)
  - 2: grounding_check -> Simple property check: entity 'penguin' exists, 1 total entities
  - 3: no_answer_path -> No direct match, no inference path found

## mi-002 - Military penguins override
- expected: YES
- observed: unknown
- proof steps:
  - 1: focus -> Found 2 relevant belief(s)
  - 2: grounding_check -> Simple property check: entity 'military' exists, 1 total entities
  - 3: no_answer_path -> No direct match, no inference path found

## mi-003 - Conflicting subclass rules
- expected: YES
- observed: unknown
- proof steps:
  - 1: focus -> Found 3 relevant belief(s)
  - 2: grounding_check -> Simple property check: entity 'bat-owl' exists, 2 total entities
  - 3: no_answer_path -> No direct match, no inference path found

## mi-004 - Contradiction from two parents
- expected: CONFLICT
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: p

## mi-005 - Priority by rank
- expected: NO
- observed: unknown
- proof steps:
  - 1: focus -> Found 2 relevant belief(s)
  - 2: grounding_check -> Simple property check: entity 'p' exists, 2 total entities
  - 3: no_answer_path -> No direct match, no inference path found

## mi-007 - Diamond inheritance with opposing defaults
- expected: NO
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: p

## mi-008 - Diamond with conflicting intermediates
- expected: CONFLICT
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: p

## mi-009 - Specificity wins over strength
- expected: swimming
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: how

## mi-010 - Conflicting deep inheritance
- expected: CONFLICT
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: p

## rel-001 - Trusted vs untrusted source
- expected: YES
- observed: unknown
- proof steps:
  - 1: entity_existence_check -> Primary entity not in knowledge base: work

