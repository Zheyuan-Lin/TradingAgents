"""The eval module's append-only store: hash chaining, per-ticker isolation,
and the read-time join across decisions/settlements/judgments/audits."""

from __future__ import annotations

import json

import pytest

from tradingagents.eval.schema import Audit, Decision, Judgment, ModelInfo, RunConfig, Settlement
from tradingagents.eval.store import EvalStore


def _model() -> ModelInfo:
    return ModelInfo(
        provider="anthropic", deep_think_llm="claude-opus-5",
        quick_think_llm="claude-sonnet-5", recorded_cutoff="2026-01",
    )


def _decision(record_id: str, ticker: str = "NVDA", **overrides) -> Decision:
    base = {
        "record_id": record_id, "ticker": ticker, "as_of_date": "2026-08-01",
        "generated_at": "2026-08-01T21:00:00Z", "asset_type": "stock",
        "selected_analysts": ["market"], "model": _model(),
        "run_config": RunConfig(debate_rounds=1, risk_rounds=1, temperature=None, prompt_hash="abc"),
        "cohort_id": "c1", "arm": "full", "run_kind": "live", "portfolio": None, "status": "ok",
    }
    base.update(overrides)
    return Decision(**base)


@pytest.mark.unit
def test_the_first_decision_for_a_ticker_has_no_prev_hash(tmp_path):
    store = EvalStore(tmp_path)
    written = store.append_decision(_decision("r1"))
    assert written.prev_hash == "" and written.record_hash != ""


@pytest.mark.unit
def test_a_second_decision_chains_to_the_first(tmp_path):
    store = EvalStore(tmp_path)
    first = store.append_decision(_decision("r1"))
    second = store.append_decision(_decision("r2"))
    assert second.prev_hash == first.record_hash


@pytest.mark.unit
def test_load_decisions_returns_what_was_appended_in_order(tmp_path):
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_decision(_decision("r2"))
    loaded = store.load_decisions("NVDA")
    assert [d.record_id for d in loaded] == ["r1", "r2"]


@pytest.mark.unit
def test_an_intact_chain_verifies_clean(tmp_path):
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_decision(_decision("r2"))
    store.append_decision(_decision("r3"))
    assert store.verify_chain("NVDA") == []


@pytest.mark.unit
def test_tampering_content_without_touching_its_hash_is_caught_on_that_record(tmp_path):
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_decision(_decision("r2"))

    path = tmp_path / "NVDA" / "decisions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["final_signal"] = "Sell"  # content changed, record_hash left stale
    lines[0] = json.dumps(tampered)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    broken = store.verify_chain("NVDA")
    # r1's own stored hash no longer matches its content: caught directly.
    # r2's prev_hash still matches r1's (unchanged) stored hash, so the link
    # itself isn't broken by this — that's a second, independent check.
    assert broken == ["r1"]


@pytest.mark.unit
def test_forging_a_hash_to_hide_tampering_breaks_the_link_to_the_next_record(tmp_path):
    """The actual property a hash chain buys you: you can forge one record's
    own hash to match tampered content, but you can't retroactively fix the
    *next* record's prev_hash without rewriting it too."""
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_decision(_decision("r2"))

    path = tmp_path / "NVDA" / "decisions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = Decision.from_dict(json.loads(lines[0]))
    tampered.final_signal = "Sell"
    tampered.record_hash = tampered.content_hash()  # forge it to look self-consistent
    lines[0] = json.dumps(tampered.to_dict())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    broken = store.verify_chain("NVDA")
    assert broken == ["r2"]  # r1 looks clean in isolation; r2's prev_hash now mismatches


@pytest.mark.unit
def test_tickers_are_stored_independently(tmp_path):
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1", ticker="NVDA"))
    store.append_decision(_decision("r2", ticker="AAPL"))
    assert [d.record_id for d in store.load_decisions("NVDA")] == ["r1"]
    assert [d.record_id for d in store.load_decisions("AAPL")] == ["r2"]


@pytest.mark.unit
def test_a_ticker_with_no_data_yet_loads_empty(tmp_path):
    store = EvalStore(tmp_path)
    assert store.load_decisions("NVDA") == []
    assert store.load_settlements("NVDA") == []
    assert store.join_records("NVDA") == []


@pytest.mark.unit
def test_settlements_judgments_and_audits_round_trip(tmp_path):
    store = EvalStore(tmp_path)
    settlement = Settlement(
        record_id="r1", horizon="5d", entry_date="2026-08-02", entry_price=100.0,
        exit_date="2026-08-09", exit_price=105.0, benchmark="SPY",
        benchmark_entry_price=500.0, benchmark_exit_price=502.0,
        raw_return=0.05, benchmark_return=0.004, alpha_return=0.046,
        price_source="yfinance", adjusted=True, settled_at="2026-08-09T21:00:00Z",
    )
    judgment = Judgment(
        record_id="r1", judge_model="gpt-5.6", prompt_version="v1", judge_kind="process",
        grounded=True, unsupported_claims=[], thesis="revenue beat drives upside",
        thesis_confirmed={"2d": None, "5d": True, "10d": None},
        debate_substantive=False, notes="", judged_at="2026-08-09T21:05:00Z",
    )
    audit = Audit(
        record_id="r1", audit_name="grounding_check_v2", result="pass",
        detail="", audited_at="2026-09-01T00:00:00Z",
    )

    store.append_settlement("NVDA", settlement)
    store.append_judgment("NVDA", judgment)
    store.append_audit("NVDA", audit)

    assert store.load_settlements("NVDA") == [settlement]
    assert store.load_judgments("NVDA") == [judgment]
    assert store.load_audits("NVDA") == [audit]


@pytest.mark.unit
def test_join_records_attaches_settlements_and_judgments_by_record_id(tmp_path):
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_decision(_decision("r2"))
    store.append_settlement("NVDA", Settlement(
        record_id="r1", horizon="2d", entry_date="2026-08-02", entry_price=100.0,
        exit_date="2026-08-04", exit_price=101.0, benchmark="SPY",
        benchmark_entry_price=500.0, benchmark_exit_price=500.5,
        raw_return=0.01, benchmark_return=0.001, alpha_return=0.009,
        price_source="yfinance", adjusted=True, settled_at="2026-08-04T21:00:00Z",
    ))

    joined = {r.decision.record_id: r for r in store.join_records("NVDA")}
    assert len(joined["r1"].settlements) == 1
    assert joined["r2"].settlements == []


@pytest.mark.unit
def test_join_records_ignores_a_settlement_with_no_matching_decision(tmp_path):
    """An orphaned settlement (e.g. from a decision in another ticker's log
    by a bad caller) must not crash the join or attach to the wrong record."""
    store = EvalStore(tmp_path)
    store.append_decision(_decision("r1"))
    store.append_settlement("NVDA", Settlement(
        record_id="does-not-exist", horizon="2d", entry_date="2026-08-02", entry_price=100.0,
        exit_date="2026-08-04", exit_price=101.0, benchmark="SPY",
        benchmark_entry_price=500.0, benchmark_exit_price=500.5,
        raw_return=0.01, benchmark_return=0.001, alpha_return=0.009,
        price_source="yfinance", adjusted=True, settled_at="2026-08-04T21:00:00Z",
    ))

    joined = store.join_records("NVDA")
    assert len(joined) == 1 and joined[0].settlements == []
