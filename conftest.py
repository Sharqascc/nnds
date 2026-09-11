

# Hypothesis CI profile: deterministic seeds for reproducible CI.
try:
    from hypothesis import settings as _hyp_settings

    _hyp_settings.register_profile("ci", derandomize=True, max_examples=50)
    _hyp_settings.load_profile("ci")
except ImportError:
    pass
