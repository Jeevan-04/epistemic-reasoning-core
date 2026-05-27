#!/usr/bin/env python3
"""Generate publication-ready figures from aggregated benchmark results.

Writes PNG files to paper/figures/:
- pipeline_localization.png
- failure_distribution.png
- mode_comparison.png

If matplotlib is not installed, this script exits with a helpful message.
"""
import json
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AGG_PATH = os.path.join(ROOT, 'tests', 'logs', 'bench_compare_results.json')
OUT_DIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        print('Matplotlib is not available:', e)
        print('To generate figures, install dependencies:')
        print('    pip install -r requirements.txt')
        return None


def plot_failure_distribution(data, out_path):
    plt = try_import_matplotlib()
    if plt is None:
        return
    fd = data.get('failure_distribution', [])
    if not fd:
        print('No failure distribution data found, skipping')
        return
    reasons = [f"{r['stage']}/{r['reason']}" for r in fd]
    counts = [r['count'] for r in fd]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(reasons, counts, color='tab:orange')
    ax.set_xlabel('Count')
    ax.set_title('Failure Distribution')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mode_comparison(data, out_path):
    plt = try_import_matplotlib()
    if plt is None:
        return
    bm = data.get('by_mode', {})
    modes = []
    ground = []
    reason = []
    for m in ['nl', 'symbolic']:
        if m in bm:
            modes.append(m)
            val_g = bm[m].get('grounding_success_pct')
            val_r = bm[m].get('reasoning_success_pct')
            ground.append(0 if val_g == 'N/A' else val_g)
            reason.append(0 if val_r == 'N/A' else val_r)
    x = range(len(modes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6,3.5))
    ax.bar([i - width/2 for i in x], ground, width, label='Ground %', color='tab:blue')
    ax.bar([i + width/2 for i in x], reason, width, label='Reason %', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel('Percent')
    ax.set_title('Mode comparison: grounding vs reasoning (percent)')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pipeline_localization(data, out_path):
    plt = try_import_matplotlib()
    if plt is None:
        return
    bm = data.get('by_mode', {})
    entries = {}
    for m, v in bm.items():
        count = v.get('count', 0)
        def pct_to_count(key):
            pct = v.get(key)
            if isinstance(pct, (int, float)):
                return int(round(count * (pct / 100.0)))
            return count
        parser_ok = pct_to_count('parse_success_pct') if v.get('parse_success_pct') != 'N/A' else count
        ground_ok = pct_to_count('grounding_success_pct')
        reason_ok = pct_to_count('reasoning_success_pct')
        entries[m] = {'parser': parser_ok, 'ground': ground_ok, 'reason': reason_ok}

    labels = ['parser', 'ground', 'reason']
    fig, ax = plt.subplots(figsize=(6,3.5))
    x = range(len(labels))
    width = 0.35
    modes = list(entries.keys())
    for i, m in enumerate(modes):
        vals = [entries[m][k] for k in labels]
        ax.bar([p + i*width for p in x], vals, width, label=m)
    ax.set_xticks([p + width*(len(modes)-1)/2 for p in x])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Count')
    ax.set_title('Pipeline localization: counts by stage')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    if not os.path.exists(AGG_PATH):
        print('Aggregated results not found at', AGG_PATH)
        sys.exit(1)
    data = load_json(AGG_PATH)
    plot_failure_distribution(data, os.path.join(OUT_DIR, 'failure_distribution.png'))
    plot_mode_comparison(data, os.path.join(OUT_DIR, 'mode_comparison.png'))
    plot_pipeline_localization(data, os.path.join(OUT_DIR, 'pipeline_localization.png'))
    print('Wrote figures to', OUT_DIR)


if __name__ == '__main__':
    main()
