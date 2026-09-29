import builtins
import importlib


def _reload_analysis():
    import src.analysis as analysis

    importlib.reload(analysis)
    return analysis


def test_analysis_init_success():
    analysis = _reload_analysis()
    assert analysis._viz_available is True
    assert analysis._pet_summary_available is True
    assert analysis.check_installation()["visualization"] is True
    assert analysis.check_installation()["pet_summary"] is True


def _load_analysis_source():
    import src.analysis as analysis

    return Path(analysis.__file__).read_text()


def _exec_analysis_with_fake_import(fake_import):
    code = _load_analysis_source()
    ns = {"__name__": "src.analysis", "__package__": "src.analysis"}
    original = builtins.__import__
    builtins.__import__ = fake_import
    try:
        exec(compile(code, "<analysis_init>", "exec"), ns)
    finally:
        builtins.__import__ = original
    return ns
