import json
import os
from datetime import datetime


def _slugify(text: str) -> str:
    return (
        text.lower()
        .replace(' ', '_')
        .replace('/', '_')
        .replace('?', '')
        .replace('\n', '')
    )


def _serialize(obj):
    # Generic serializer: try json-serializable, then dataclass-like, else repr
    try:
        json.dumps(obj)
        return obj
    except Exception:
        pass
    if hasattr(obj, '__dict__'):
        out = {}
        for k, v in vars(obj).items():
            out[k] = _serialize(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def save_proof(proof, scenario_name: str, verdict: str, out_dir='tests/logs', details=None):
    os.makedirs(out_dir, exist_ok=True)
    slug = _slugify(scenario_name)[:120]
    filename = f"proof_{slug}_{int(datetime.utcnow().timestamp())}.json"
    path = os.path.join(out_dir, filename)
    data = {
        'scenario': scenario_name,
        'verdict': verdict,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'proof': _serialize(proof),
    }
    if details is not None:
        data['details'] = _serialize(details)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    return path
