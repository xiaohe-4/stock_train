import json
import shutil

from research.reliability.fetch_pit import fetch_pit, locate_reusable, resolve_pit
from research.reliability.strict_run import run_strict
from tests.test_reliability_fetch import FlakyClient


def _store(tmp_path, name="pit"):
    dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    root = tmp_path / name
    fetch_pit(root, client, start=dates[0], end=dates[-1], constituents_per_day=2, sleep=lambda _seconds: None, retries=3)
    return root, dates, client


def test_verified_cache_is_reused_without_calling_the_client(tmp_path):
    root, dates, _client = _store(tmp_path)
    cached = tmp_path / "input" / "hs300-reliability-evaluation" / "hs300_reliability" / "pit"
    shutil.copytree(root, cached)
    before = (cached / "manifest.json").read_bytes()
    found, reason = locate_reusable([tmp_path / "input"], dates[0], dates[-1])
    assert reason == ""
    assert found == cached
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    resolved = resolve_pit(
        tmp_path / "fresh",
        client,
        search_roots=[tmp_path / "input"],
        start=dates[0],
        end=dates[-1],
        constituents_per_day=2,
        sleep=lambda _seconds: None,
        retries=1,
    )
    assert resolved["data_source"] == "reused"
    assert resolved["data_dir"] == str(cached)
    assert client.constituent_calls == 0
    assert client.bars_calls == 0
    assert (cached / "manifest.json").read_bytes() == before
    assert not (tmp_path / "fresh").exists()


def test_hash_or_range_mismatch_downloads_instead(tmp_path):
    root, dates, _client = _store(tmp_path)
    cached = tmp_path / "input" / "hs300_reliability" / "pit"
    shutil.copytree(root, cached)
    bar = next((cached / "bars").glob("*.csv"))
    bar.write_text(bar.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    found, reason = locate_reusable([tmp_path / "input"], dates[0], dates[-1])
    assert found is None
    assert "sha256" in reason
    client = FlakyClient(dates, ["sh.600000", "sh.600132"])
    resolved = resolve_pit(
        tmp_path / "fresh",
        client,
        search_roots=[tmp_path / "input"],
        start=dates[0],
        end=dates[-1],
        constituents_per_day=2,
        sleep=lambda _seconds: None,
        retries=3,
    )
    assert resolved["data_source"] == "downloaded"
    assert client.constituent_calls > 0
    manifest = json.loads((tmp_path / "fresh" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_source"] == "downloaded"
    short, short_reason = locate_reusable([tmp_path / "fresh"], "2024-01-02", "2024-01-08")
    assert short is None
    assert "date range" in short_reason


def test_reused_data_still_fails_closed_when_the_audit_fails(tmp_path):
    root, dates, _client = _store(tmp_path, "source")
    cached = tmp_path / "input" / "hs300_reliability" / "pit"
    shutil.copytree(root, cached)
    next((cached / "bars").glob("*.csv")).unlink()
    result = run_strict(
        cached,
        tmp_path / "out",
        expected_days=3,
        constituents_per_day=2,
        start=dates[0],
        end=dates[-1],
        data_source="reused",
        data_source_path=str(cached),
        reuse_rejection=None,
    )
    assert result["trained"] is False
    assert result["strict"] is False
    assert result["status"] == "fail"
    assert result["data_source"] == "reused"
    report = json.loads((tmp_path / "out" / "audit_report.json").read_text(encoding="utf-8"))
    assert report["strict_ready"] is False
    assert report["data_source"] == "reused"
    assert report["data_source_path"] == str(cached)
