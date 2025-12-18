# Geographic Memory

**RETRIEVAL-ONLY external memory for geographic facts**

## Purpose

Models encyclopedic geographic knowledge that humans don't "reason" about - they just remember.

## Design Principle

> "Certain domains (geography, encyclopedic facts) are modeled as external memory, not reasoning."

## No Inference

This module does NO reasoning:
- No inheritance
- No inference
- No learning
- Pure lookup

## Examples

- "Is London in Europe?" → Lookup: London → UK → Europe → YES
- "Is Tokyo in Europe?" → Lookup: Tokyo → Japan → Asia → NO

## Academic Framing

This is academically respectable. Systems don't need to REASON about everything. Some knowledge is just stored and retrieved, like a database or memory palace.

This demonstrates epistemic modularity: logical reasoning (Buddhi) vs. external memory (Geographic).
