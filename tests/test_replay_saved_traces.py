import subprocess
import glob
import sys
import json


def test_replay_all_saved_traces():
    """Replay every saved proof trace with `replay_trace.py` and assert VERIFIED."""
    raw_traces = glob.glob('tests/logs/proof_*.json') + glob.glob('tests/logs/bench_*.json')
    # Filter to only those traces that look like proof traces with teachings
    traces = []
    for t in raw_traces:
        try:
            j = json.loads(open(t, 'r', encoding='utf8').read())
            if isinstance(j, dict) and isinstance(j.get('details'), dict) and j['details'].get('teachings'):
                traces.append(t)
        except Exception:
            # skip non-JSON or unexpected formats
            continue

    assert traces, "No saved, replayable traces found in tests/logs/ (skipping list may include non-trace files)"

    failures = []
    for t in traces:
        proc = subprocess.run([sys.executable, 'replay_trace.py', t], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        if 'VERIFIED' not in out:
            failures.append({'trace': t, 'exit': proc.returncode, 'out': out.splitlines()[-10:]})

    if failures:
        # Provide concise failure info for CI
        msgs = []
        for f in failures:
            msgs.append(f"{f['trace']}: exit={f['exit']} last_lines={f['out']}")
        raise AssertionError('Replay mismatches or errors:\n' + '\n'.join(msgs))
