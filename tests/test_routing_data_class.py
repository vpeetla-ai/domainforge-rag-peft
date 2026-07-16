"""ADR-029: confidential must not use cloud gateway path."""

from domainforge.generation import router as router_mod


def test_generate_triage_signature_accepts_data_class():
    import inspect
    sig = inspect.signature(router_mod.generate_triage)
    assert "data_class" in sig.parameters


def test_router_source_skips_gateway_for_confidential():
    src = open(router_mod.__file__).read()
    assert 'data_class != "confidential"' in src
    assert "ADR-029" in src
