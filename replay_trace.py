"""Replay a saved proof trace by rebuilding the reasoning pipeline.

Usage:
    python3 replay_trace.py tests/logs/proof_is_nixon_a_pacifist_1779187650.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahankara.layer import Ahankara
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
from manas.layer import Manas


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_verdict(value: Any) -> str:
    if value is None:
        return 'UNKNOWN'
    return str(value).strip().upper()


def _format_argument(argument: dict[str, Any], index: int) -> str:
    claim = str(argument.get('claim', f'argument_{index}'))
    sign = 'NOT ' if argument.get('is_negative') else ''
    meta_bits = []
    for key in ['specificity', 'rank', 'activation', 'source_reliability']:
        if key in argument:
            meta_bits.append(f'{key}={argument.get(key)}')
    if 'is_negative' in argument:
        meta_bits.append(f'is_negative={argument.get("is_negative")}')
    meta = ', '.join(meta_bits)
    line = f'  A{index}: {sign}{claim}'
    if meta:
        line += f' [{meta}]'
    path = argument.get('path') or argument.get('supports')
    if isinstance(path, list) and path:
        line += f"\n      path: {' -> '.join(str(item) for item in path)}"
    return line


def _attack_graph_lines(arguments: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, argument in enumerate(arguments, start=1):
        if not isinstance(argument, dict):
            continue
        claim = str(argument.get('claim', '')).strip().lower()
        if claim:
            groups[claim].append((index, argument))

    for claim, items in groups.items():
        positives = [index for index, arg in items if not arg.get('is_negative')]
        negatives = [index for index, arg in items if arg.get('is_negative')]
        if positives and negatives:
            lines.append(f'  {claim}: A{positives[0]} <-> A{negatives[0]}')
    return lines


def _defeat_summary(arguments: list[dict[str, Any]]) -> list[str]:
    conflict = next((argument for argument in arguments if isinstance(argument, dict) and argument.get('winner') == 'conflict'), None)
    if not conflict:
        return []

    positive_path = conflict.get('positive_path', []) if isinstance(conflict.get('positive_path'), list) else []
    negative_path = conflict.get('negative_path', []) if isinstance(conflict.get('negative_path'), list) else []
    specificity = conflict.get('specificity', '?')
    positive_rank = None
    negative_rank = None
    positive_activation = None
    negative_activation = None
    positive_reliability = None
    negative_reliability = None

    positive_arg = next((argument for argument in arguments if isinstance(argument, dict) and not argument.get('is_negative')), None)
    negative_arg = next((argument for argument in arguments if isinstance(argument, dict) and argument.get('is_negative')), None)
    if positive_arg:
        positive_rank = positive_arg.get('rank', '?')
        positive_activation = positive_arg.get('activation', '?')
        positive_reliability = positive_arg.get('source_reliability', '?')
    if negative_arg:
        negative_rank = negative_arg.get('rank', '?')
        negative_activation = negative_arg.get('activation', '?')
        negative_reliability = negative_arg.get('source_reliability', '?')

    lines = ['Defeat ordering:']
    lines.append(f'  specificity: {specificity} vs {specificity} (tie)')
    if positive_rank is not None or negative_rank is not None:
        lines.append(f'  rank: {positive_rank} vs {negative_rank}')
    if positive_activation is not None or negative_activation is not None:
        lines.append(f'  activation: {positive_activation} vs {negative_activation}')
    if positive_reliability is not None or negative_reliability is not None:
        lines.append(f'  source_reliability: {positive_reliability} vs {negative_reliability}')
    if positive_path and negative_path:
        positive_path_text = ' -> '.join(str(item) for item in positive_path)
        negative_path_text = ' -> '.join(str(item) for item in negative_path)
        lines.append(f'  positive path: {positive_path_text}')
        lines.append(f'  negative path: {negative_path_text}')
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description='Replay a saved Episteme proof trace')
    parser.add_argument('trace_path', type=Path, help='Path to a proof_*.json trace file')
    args = parser.parse_args()

    trace_path: Path = args.trace_path
    trace = _load_json(trace_path)

    details = trace.get('details', {}) if isinstance(trace.get('details'), dict) else {}
    teachings = details.get('teachings') if isinstance(details.get('teachings'), list) else []
    query = details.get('query') or trace.get('scenario') or trace_path.stem

    if not teachings:
        raise SystemExit(f'No teachings found in {trace_path}')

    chitta = ChittaGraph()
    manas = Manas(llm_backend='mock')
    buddhi = Buddhi(chitta)
    ahankara = Ahankara(manas, buddhi, chitta)

    for teaching in teachings:
        ahankara.process(teaching)

    parsed_query = manas.parse(query)
    replay_proof = buddhi.answer(parsed_query)

    stored_proof = trace.get('proof', {}) if isinstance(trace.get('proof'), dict) else {}
    stored_verdict = _normalize_verdict(trace.get('verdict', stored_proof.get('verdict')))
    replay_verdict = _normalize_verdict(replay_proof.verdict)
    arguments = stored_proof.get('metadata', {}).get('arguments', []) if isinstance(stored_proof.get('metadata'), dict) else []

    print('TRACE REPLAY')
    print('=' * 78)
    print(f'Trace: {trace_path}')
    print(f'Query: {query}')
    print(f'Stored verdict: {stored_verdict}')
    print(f'Replayed verdict: {replay_verdict}')
    print()

    print('Teachings:')
    for teaching in teachings:
        print(f'  - {teaching}')
    print()

    print('Reconstructed arguments:')
    concrete_arguments = [argument for argument in arguments if isinstance(argument, dict) and argument.get('claim')]
    if concrete_arguments:
        for index, argument in enumerate(concrete_arguments, start=1):
            print(_format_argument(argument, index))
    else:
        print('  - none recorded')
    print()

    print('Attack graph:')
    attack_lines = _attack_graph_lines(concrete_arguments)
    if attack_lines:
        for line in attack_lines:
            print(line)
    else:
        print('  - no explicit attacks recorded')
    print()

    defeat_lines = _defeat_summary(arguments)
    if defeat_lines:
        for line in defeat_lines:
            print(line)
        print()

    print('Replay comparison:')
    if replay_verdict == stored_verdict:
        print('  VERIFIED')
    else:
        print('  MISMATCH')
    stored_steps = len(stored_proof.get('steps', [])) if isinstance(stored_proof.get('steps'), list) else 0
    print(f'  Stored steps: {stored_steps}')
    print(f'  Replayed steps: {len(replay_proof.steps)}')
    print(f'  Replayed conflicts: {len(replay_proof.conflicts)}')

    return 0 if replay_verdict == stored_verdict else 1


if __name__ == '__main__':
    raise SystemExit(main())