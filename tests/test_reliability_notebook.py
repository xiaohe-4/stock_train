import importlib.util
from pathlib import Path


def _notebook_text():
    path = Path("kaggle/generate_reliability_notebook.py")
    spec = importlib.util.spec_from_file_location("generate_reliability_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = module.build("1234567890abcdef")
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_reliability_notebook_fetches_then_trains_only_after_audit():
    text = _notebook_text()
    assert 'PINNED_COMMIT = "1234567890abcdef"' in text
    assert "research.reliability.fetch_pit" in text
    assert "research.reliability.strict_run" in text
    assert "--audit-only" in text
    assert "hs300_daily_hfq" not in text
    assert "baostock==0.9.40" not in text
    assert "importlib.reload" not in text
    assert "--full" not in text
    assert "feature_cols_map" not in text
    assert "--reuse-search" in text
    assert "/kaggle/input" in text
    assert "DATA_SOURCE" in text
    assert "2024" in text
