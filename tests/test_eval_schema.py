"""Round-trip serialization and hash-chain content hashing for the eval schema."""

from __future__ import annotations

import pytest

from tradingagents.eval.schema import (
    Audit,
    Decision,
    Judgment,
    ModelInfo,
    RunConfig,
    Settlement,
    Trajectory,
)


def _model() -> ModelInfo:
    return ModelInfo(
        provider="anthropic", deep_think_llm="claude-opus-5",
        quick_think_llm="claude-sonnet-5", recorded_cutoff="2026-01",
    )


def _run_config() -> RunConfig:
    return RunConfig(debate_rounds=1, risk_rounds=1, temperature=None, prompt_hash="abc123")


def _decision(**overrides) -> Decision:
    base = {
        "record_id": "r1", "ticker": "NVDA", "as_of_date": "2026-08-01",
        "generated_at": "2026-08-01T21:00:00Z", "asset_type": "stock",
        "selected_analysts": ["market"], "model": _model(), "run_config": _run_config(),
        "cohort_id": "c1", "arm": "full", "run_kind": "live", "portfolio": None, "status": "ok",
    }
    base.update(overrides)
    return Decision(**base)


@pytest.mark.unit
def test_decision_round_trips_through_dict_without_trajectory():
    d = _decision()
    assert Decision.from_dict(d.to_dict()) == d


@pytest.mark.unit
def test_decision_round_trips_with_a_trajectory():
    trajectory = Trajectory(
        reports={"market_report": "x"},
        debate={"investment_debate_state": {"judge_decision": "y"}},
        trader_investment_plan="plan",
        raw_tool_calls=[{"tool_name": "get_stock_data", "args": {}, "content": "...", "ts": "t"}],
    )
    d = _decision(trajectory=trajectory)
    assert Decision.from_dict(d.to_dict()) == d


@pytest.mark.unit
def test_a_failed_decision_has_no_trajectory():
    d = _decision(status="failed", error="vendor exploded", raw_signal=None, final_signal=None)
    round_tripped = Decision.from_dict(d.to_dict())
    assert round_tripped.trajectory is None
    assert round_tripped.status == "failed" and round_tripped.error == "vendor exploded"


@pytest.mark.unit
def test_content_hash_is_deterministic():
    assert _decision().content_hash() == _decision().content_hash()


@pytest.mark.unit
def test_content_hash_changes_when_content_changes():
    assert _decision().content_hash() != _decision(final_signal="Sell").content_hash()


@pytest.mark.unit
def test_content_hash_excludes_the_chain_fields_themselves():
    """The chain fields point at other records, not this one's content —
    setting them must not change what this record hashes to."""
    d = _decision()
    before = d.content_hash()
    d.prev_hash = "some-prior-hash"
    d.record_hash = "whatever"
    assert d.content_hash() == before


@pytest.mark.unit
def test_settlement_round_trips_through_dict():
    s = Settlement(
        record_id="r1", horizon="5d", entry_date="2026-08-02", entry_price=100.0,
        exit_date="2026-08-09", exit_price=105.0, benchmark="SPY",
        benchmark_entry_price=500.0, benchmark_exit_price=502.0,
        raw_return=0.05, benchmark_return=0.004, alpha_return=0.046,
        price_source="yfinance", adjusted=True, settled_at="2026-08-09T21:00:00Z",
    )
    assert Settlement.from_dict(s.to_dict()) == s


@pytest.mark.unit
def test_judgment_round_trips_through_dict():
    j = Judgment(
        record_id="r1", judge_model="gpt-5.6", prompt_version="v1", judge_kind="process",
        grounded=True, unsupported_claims=[], thesis="revenue beat drives upside",
        thesis_confirmed={"2d": None, "5d": True, "10d": None},
        debate_substantive=False, notes="bear conceded without engaging the guidance cut",
        judged_at="2026-08-09T21:05:00Z",
    )
    assert Judgment.from_dict(j.to_dict()) == j


@pytest.mark.unit
def test_audit_round_trips_through_dict():
    a = Audit(
        record_id="r1", audit_name="grounding_check_v2", result="flagged",
        detail="cited 14.7% not found in raw_tool_calls", audited_at="2026-09-01T00:00:00Z",
    )
    assert Audit.from_dict(a.to_dict()) == a
