# Algorithmic Specification Draft

This document makes the current target algorithms explicit enough to test and
criticize. It is intentionally conservative: when the implementation does not
yet satisfy the specification, the gap should become a tracked research task.

## Symbols

Let:

- `n = |B_t|`, the number of active beliefs.
- `m = |E_t|`, the number of stored graph edges.
- `d`, the maximum taxonomic depth explored.
- `a`, the number of constructed arguments for a query.
- `k`, the maximum support size of an argument.

All current reasoning is over a finite in-memory graph, so query answering is
decidable under bounded parsing and finite traversal.

## Query Algorithm

```text
ANSWER(query):
  proposal := PARSE(query)
  if malformed(proposal): return INVALID

  focus := FOCUS(B_t, proposal.entities, proposal.predicates)
  if focus is empty: return UNKNOWN

  arguments_pos := BUILD_ARGUMENTS(proposal.literal, focus, B_t)
  arguments_neg := BUILD_ARGUMENTS(negate(proposal.literal), focus, B_t)

  attacks := BUILD_ATTACKS(arguments_pos, arguments_neg)
  undefeated := RESOLVE_DEFEATS(arguments_pos ∪ arguments_neg, attacks)

  return VERDICT(undefeated, proposal.literal)
```

## Argument Construction

```text
BUILD_ARGUMENTS(literal L, focus, B_t):
  args := []

  for belief b in focus:
    if directly_supports(b, L):
      args.append(<L, [b], direct>)

  for entity x in L.entities:
    for taxonomic path x -> ... -> ancestor up to depth d:
      for belief b in B_t:
        if inheritable_support(b, ancestor, L):
          args.append(<L, path + [b], taxonomic>)

  for transitive relation r in L.predicates:
    for path connecting L.subject to L.object by r:
      args.append(<L, path, transitive>)

  return args
```

Expected worst-case complexity before indexing improvements:

```text
O(n + d*n + m)
```

The current implementation still performs several scans over all beliefs. A
publishable version should report both naive and indexed complexity.

## Attack Construction

```text
BUILD_ATTACKS(args):
  for each pair (Ai, Aj):
    if complementary(claim(Ai), claim(Aj)):
      add mutual attack
    if path_blocked(Ai, Aj):
      add directed attack from blocker to blocked argument
```

Naive complexity:

```text
O(a^2)
```

Indexed by predicate and entity, this can be reduced to comparisons among
arguments with compatible signatures.

## Defeat Resolution

For each attack pair, compare arguments lexicographically:

```text
specificity > epistemic_rank > source_reliability > activation > recency
```

If neither argument strictly defeats the other, both remain undefeated and the
verdict can become `CONFLICT`.

Naive complexity:

```text
O(a^2 * k)
```

## Decidability

For finite `B_t`, finite arity predicates, and bounded traversal depth or
cycle-checked graph traversal, query answering terminates. The current
implementation uses finite loops and visited sets for taxonomic traversal, but
the final paper still needs a formal decidability proof tied to the exact
implemented algorithms.

## Known Implementation Gaps

- The current code does not yet expose a complete argument object for every
  verdict.
- Attack/defeat records are partial and mostly embedded in proof steps.
- Nixon-style conflict is not yet robust in the seed research benchmark.
- Source reliability is stored as a field but not yet used in defeat ordering.
- Activation exists as a compatibility field but does not yet drive an ablation
  with independent behavior.

## 7. Precise Pseudocode: Defeat Resolution and Verdict

RESOLVE_DEFEATS implements the lexicographic defeat ordering described in the
formal core. It receives two sets of arguments (positive and negative) and a
set of attack relations.

```text
RESOLVE_DEFEATS(args, attacks):
  undefeated := set(args)

  # Process all attack edges (directed: attacker -> attacked)
  for (A -> B) in attacks:
    if A not in undefeated or B not in undefeated: continue

    if A.defeats(B):
      # A strictly defeats B -> remove B from undefeated
      undefeated.remove(B)
    elif B.defeats(A):
      undefeated.remove(A)
    else:
      # Neither strictly defeats the other -> both survive (possible conflict)
      continue

  return undefeated
```

After `undefeated` is produced, the `VERDICT` function inspects whether there
exist undefeated arguments for $L$ and $\lnot L$ and maps to {YES, NO,
CONFLICT, UNKNOWN} as in the formal core.

## 8. Complexity Discussion (expanded)

Naive costs assume no indexing and worst-case branching.

- Let $n=|B_t|$ (active beliefs), $d$ maximum taxonomic depth explored,
  $b$ average branching factor of the taxonomic graph, and $a$ number of
  constructed arguments for the query.

- Argument construction (taxonomic): exploring depth $d$ with branching $b$ can
  yield up to $O(b^d)$ candidate paths in pathological graphs. For each path we
  scan supporting beliefs leading to $O(n \cdot b^d)$ worst-case cost.

- Building attacks naively compares all pairs of arguments: $O(a^2\cdot k)$
  where $k$ is support size used in argument comparisons.

- Defeat resolution on the attack graph is $O(|E_{attacks}|)$ where each edge
  comparison is $O(k)$ for aggregating vector components; with $|E_{attacks}|=O(a^2)$
  we obtain $O(a^2\cdot k)$.

Indexed costs: index by predicate, subject entity, and taxonomic ancestors:

- Focus extraction: $O(1)$ to find candidate beliefs via inverted indices.
- Argument construction limited to relevant beliefs reduces $n$ to $n_f$ (focus size).
- Attack construction limited to arguments sharing predicate/subject signatures reduces pairwise checks to a small neighborhood proportional to $a_{local}^2$.

In practice the two key mitigations are:
1. Early focus filtering via inverted indices on `(predicate, subject)` to bound $n_f\ll n`.
2. Cycle-checked traversal with depth limits to prevent exponential taxonomic expansion.

Empirical evaluation should report both naive theoretical bounds and measured
scaling curves for: varying $n$, varying $d$, varying $b$, and varying query fanout
to capture argument explosion. The final paper should include both asymptotic
derivations and microbenchmarks showing where indices reduce practical costs.

