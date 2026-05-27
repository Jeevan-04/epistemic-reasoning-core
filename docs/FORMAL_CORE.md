# Formal Core Draft

This document defines the formal target for the Episteme rebuild. It is a
working specification, not yet a completed proof system.

## 1. Language

Let:

- `E` be a finite set of entity symbols.
- `P` be a finite set of predicate symbols.
- A literal be either `p(e1, ..., en)` or `not p(e1, ..., en)`.
- A taxonomic assertion be represented as `is_a(subject, object)`.
- A belief record be:

```text
b = <id, literal, epistemic_rank, activation, support_count,
     source_reliability, timestamp, provenance>
```

Current verdicts are:

```text
V = {YES, NO, UNKNOWN, CONFLICT, INVALID}
```

## 2. Active Belief Set

At time `t`, the active belief set is:

```text
B_t = { b | activation_t(b) >= theta_on or b.active = true }
```

For hysteresis-based availability, a belief remains active while:

```text
activation_t(b) > theta_off
```

with:

```text
theta_on > theta_off
```

This prevents rapid active/inactive oscillation near one fixed threshold.

## 3. Argument Construction

An argument for a query literal `L` is a finite support set:

```text
A = <claim=L, supports=[b1, ..., bn], path=[r1, ..., rk]>
```

Arguments may be:

- direct: a belief directly asserts `L`
- taxonomic: `x is_a y` and an inheritable property applies to `y`
- transitive: a transitive relation path connects query endpoints

An argument is grounded only if every support belief is in `B_t`.

## 4. Attack Relation

Two arguments attack each other when their claims are complementary:

```text
claim(A) = L
claim(B) = not L
```

or when one argument depends on a taxonomic path explicitly blocked by a more
specific negative belief.

## 5. Defeat Ordering

For competing arguments, Episteme uses the following lexicographic ordering:

```text
specificity > epistemic_rank > source_reliability > activation > recency
```

Specificity is measured by path distance from the query subject. Shorter
applicable paths are more specific.

Epistemic rank is currently:

```text
AXIOM > OBSERVATION > EXCEPTION > DEFAULT > HYPOTHESIS > UNKNOWN
```

This ordering is a research hypothesis and must be tested by ablation.

## 6. Verdict Semantics

For query `Q`:

- `YES`: at least one undefeated argument supports `Q`, and no undefeated
  argument supports `not Q`.
- `NO`: at least one undefeated argument supports `not Q`, and no undefeated
  argument supports `Q`.
- `CONFLICT`: undefeated arguments support both `Q` and `not Q`.
- `UNKNOWN`: no grounded undefeated argument supports either `Q` or `not Q`.
- `INVALID`: parser or schema validation fails before reasoning.

## 7. Required Proof Obligations

Before publication, the system needs:

- decidability proof for finite `B_t` and bounded relation arity
- complexity analysis for direct, taxonomic, and attack-resolution reasoning
- soundness of proof trace replay relative to verdict semantics
- explicit failure mode taxonomy
- ablation evidence for activation and defeat-order components

## 8. Formal Argument Object, Inference Rules, and Defeat Ordering

### 8.1 Formal Argument Object

We make the argument object explicit for algorithms and proofs. An argument is a tuple

$$
A = \langle claim(L), supports(S), path(\pi), rank(r),
        specificity(s), source\_reliability(\rho), activation(\alpha), recency(\tau), provenance(P)\rangle
$$

where $S$ is the finite set of supporting beliefs, $\pi$ is the taxonomic or
relation path (an ordered list), $r$ is an epistemic rank on a totally ordered
scale, $s\in\mathbb{N}$ is a distance metric (lower = more specific),
$\rho,\alpha\in[0,1]$ and $\tau$ is a monotone timestamp.

### 8.2 Inference Rules (construction)

- Direct rule: if $b\in B_t$ and $literal(b)=L$ then create a direct
  argument with $supports=\{b\}$ and $s=0$.
- Taxonomic inheritance: given taxonomic chain $x\to\dots\to y$ and an
  inheritable property of $y$, produce an argument whose $s$ is path length.
- Transitive composition: compose relation edges matching query predicate to
  produce a transitive argument. Each rule must specify deterministic
  aggregation functions for $r,\rho,\alpha,\tau$ so arguments are comparable.

### 8.3 Defeat Ordering (Formal)

Let $v(A) = (-s(A), r(A), \rho(A), \alpha(A), \tau(A))$. Compare $v(A)$ and
$v(B)$ lexicographically (larger vector is stronger). $A$ strictly defeats $B$
iff $v(A)>v(B)$. If neither wins then both remain undefeated (horizontal
conflict -> `CONFLICT`).

This explicit vectorization clarifies where indices and optimizations may be
inserted and makes ablation straightforward: remove or reweight components
and observe behavior change.

