import re
from pathlib import Path

import pytest

from buddhi.layer import Buddhi

# Record proofs produced during a single test run
_LAST_PROOFS = []

# Wrap Buddhi.answer to capture AnswerProof objects
_orig_answer = Buddhi.answer


def _answer_and_record(self, proposal):
    proof = _orig_answer(self, proposal)
    try:
        _LAST_PROOFS.append((proposal, proof))
    except Exception:
        pass
    return proof


Buddhi.answer = _answer_and_record


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _LAST_PROOFS.clear()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # Execute other hooks first
    outcome = yield
    rep = outcome.get_result()
    if rep.when == 'call' and rep.failed:
        # Save all proofs collected during this test
        from tests.utils.save_proof import save_proof
        node = re.sub(r'[^A-Za-z0-9_.-]', '_', item.nodeid)
        out_dir = Path('tests/logs/pytest')
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (proposal, proof) in enumerate(list(_LAST_PROOFS)):
            name = f"{node}_proof_{i}"
            try:
                save_proof(proof, name, str(proof.verdict).upper(), out_dir=str(out_dir))
            except Exception:
                pass
