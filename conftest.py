"""Pytest configuration: Hypothesis profiles and test classification."""

from __future__ import annotations

import os

# Hypothesis profiles: different scales for different execution contexts.
#
#   ci       (default)  50 examples  - pre-push, local developer runs
#   ci-pr            300 examples  - PR gate (CI workflow, `ci.yml`)
#   ci-deep         2500 examples  - nightly deep verification
#
# Select via HYPOTHESIS_PROFILE env var. Unset -> "ci", so local runs
# are unchanged. All profiles derandomize for reproducible results.
try:
    from hypothesis import settings as _hyp_settings

    _profile = os.environ.get("HYPOTHESIS_PROFILE", "ci")
    _hyp_settings.register_profile("ci", derandomize=True, max_examples=50)
    _hyp_settings.register_profile("ci-pr", derandomize=True, max_examples=300, deadline=2000)
    _hyp_settings.register_profile("ci-deep", derandomize=True, max_examples=2500, deadline=None)
    _hyp_settings.load_profile(_profile)
except ImportError:
    pass


def pytest_collection_modifyitems(config, items):
    """Auto-apply @pytest.mark.property based on filename convention.

    Convention (covers 38 of 42 property files):
      - files ending in `_property.py`
      - files containing `property_based` in the name

    The remaining 3 files use @given without matching the convention
    (test_agentic_fix.py, test_agentic_fix_ollama.py, test_ssm_technical.py)
    and carry explicit @pytest.mark.property decorators on each test.

    Rationale: keep property-test classification automatic so new files
    stay consistent without authors needing to remember.
    """
    import pytest

    for item in items:
        if "property" in item.keywords:
            continue
        fspath = str(item.fspath)
        if fspath.endswith("_property.py") or "property_based" in fspath:
            item.add_marker(pytest.mark.property)


@pytest.fixture
def seeded_rng():
    """Yield a scoped numpy Generator so tests do not touch global RNG state.

    Usage:
        def test_foo(seeded_rng):
            x = seeded_rng.normal(size=10)

    The default seed is fixed (1234) so failing tests are reproducible.
    Override per test by constructing your own default_rng(seed).
    """
    import numpy as np

    return np.random.default_rng(1234)
