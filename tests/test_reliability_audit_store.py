import json

from research.reliability.audit import audit_store
from research.reliability.fetch_pit import fetch_pit
from tests.test_reliability_fetch import FlakyClient


def _store(tmp_path):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    root = tmp_path / "pit"
    fetch_pit(root, client, start=dates[0], end=dates[-1], constituents_per_day=2, sleep=lambda _seconds: None, retries=3)
    return root, dates


def test_complete_store_is_strict(tmp_path):
    root, dates = _store(tmp_path)
    audit = audit_store(root, expected_days=3, constituents_per_day=2, start=dates[0], end=dates[-1])
    assert audit["strict_ready"] is True
    assert audit["mode"] == "strict"
    assert audit["blockers"] == []


def test_missing_bars_fail_closed(tmp_path):
    root, dates = _store(tmp_path)
    next((root / "bars").glob("*.csv")).unlink()
    audit = audit_store(root, expected_days=3, constituents_per_day=2, start=dates[0], end=dates[-1])
    assert audit["strict_ready"] is False
    assert audit["mode"] == "research"
    assert any("不复权" in item or "缺失" in item for item in audit["blockers"])


def test_hash_mismatch_fails(tmp_path):
    root, dates = _store(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = "membership/2024-01-02.json"
    manifest["files"][key]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit = audit_store(root, expected_days=3, constituents_per_day=2, start=dates[0], end=dates[-1])
    assert audit["strict_ready"] is False
    assert any("sha256" in item for item in audit["blockers"])
