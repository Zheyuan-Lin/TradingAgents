"""Typed records for the eval module's four append-only logs.

See docs/eval-framework-design.md §4. Nothing here is ever mutated after it's
written to a log — a correction is a new row (an Audit), not an edit to an
existing Decision/Settlement/Judgment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class ModelInfo:
    provider: str
    deep_think_llm: str
    quick_think_llm: str
    recorded_cutoff: str  # audit trail — changes when the model changes


@dataclass
class RunConfig:
    debate_rounds: int
    risk_rounds: int
    temperature: float | None
    prompt_hash: str


@dataclass
class Trajectory:
    reports: dict[str, str]
    debate: dict[str, dict]
    trader_investment_plan: str
    raw_tool_calls: list[dict]  # extracted from final_state["messages"]


@dataclass
class Decision:
    """One generation attempt, including failures. Never rewritten after
    ``EvalStore.append_decision`` fills in its hash-chain fields and writes it."""

    record_id: str
    ticker: str
    as_of_date: str  # trading day whose post-close data was used
    generated_at: str
    asset_type: str
    selected_analysts: list[str]
    model: ModelInfo
    run_config: RunConfig
    cohort_id: str
    arm: str  # ablation label: "full" | "no_debate" | "no_memory" | ...
    run_kind: str  # "live" | "redteam" | "canary"
    portfolio: dict | None
    status: str  # "ok" | "failed" | "review"
    error: str | None = None
    raw_signal: str | None = None  # before any guardrail rewrite
    final_signal: str | None = None  # after guardrails — never overwrite raw_signal
    final_trade_decision: str | None = None
    guardrail_events: list[dict] = field(default_factory=list)
    trajectory: Trajectory | None = None
    prev_hash: str = ""
    record_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "generated_at": self.generated_at,
            "asset_type": self.asset_type,
            "selected_analysts": self.selected_analysts,
            "model": asdict(self.model),
            "run_config": asdict(self.run_config),
            "cohort_id": self.cohort_id,
            "arm": self.arm,
            "run_kind": self.run_kind,
            "portfolio": self.portfolio,
            "status": self.status,
            "error": self.error,
            "raw_signal": self.raw_signal,
            "final_signal": self.final_signal,
            "final_trade_decision": self.final_trade_decision,
            "guardrail_events": self.guardrail_events,
            "trajectory": asdict(self.trajectory) if self.trajectory else None,
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Decision:
        trajectory = d.get("trajectory")
        return cls(
            record_id=d["record_id"],
            ticker=d["ticker"],
            as_of_date=d["as_of_date"],
            generated_at=d["generated_at"],
            asset_type=d["asset_type"],
            selected_analysts=d["selected_analysts"],
            model=ModelInfo(**d["model"]),
            run_config=RunConfig(**d["run_config"]),
            cohort_id=d["cohort_id"],
            arm=d["arm"],
            run_kind=d["run_kind"],
            portfolio=d.get("portfolio"),
            status=d["status"],
            error=d.get("error"),
            raw_signal=d.get("raw_signal"),
            final_signal=d.get("final_signal"),
            final_trade_decision=d.get("final_trade_decision"),
            guardrail_events=d.get("guardrail_events", []),
            trajectory=Trajectory(**trajectory) if trajectory else None,
            prev_hash=d.get("prev_hash", ""),
            record_hash=d.get("record_hash", ""),
        )

    def content_hash(self) -> str:
        """Hash of this record's own content, excluding the chain fields
        themselves — those point at other records, not this one's content."""
        payload = self.to_dict()
        payload.pop("prev_hash", None)
        payload.pop("record_hash", None)
        canonical = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Settlement:
    """One (record_id, horizon) outcome, written once resolved."""

    record_id: str
    horizon: str  # "2d" | "5d" | "10d" — trading days
    entry_date: str  # as_of_date's next trading day
    entry_price: float  # that day's open — not as_of_date's close; see design doc §2
    exit_date: str
    exit_price: float
    benchmark: str
    benchmark_entry_price: float
    benchmark_exit_price: float
    raw_return: float
    benchmark_return: float
    alpha_return: float
    price_source: str
    adjusted: bool  # split/dividend-adjusted prices move retroactively
    settled_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Settlement:
        return cls(**d)


@dataclass
class Judgment:
    """One judge pass over a settled record."""

    record_id: str
    judge_model: str
    prompt_version: str
    judge_kind: str  # "process" (blind to outcome) | "thesis" (outcome-aware)
    grounded: bool | None
    unsupported_claims: list[str]
    thesis: str
    thesis_confirmed: dict[str, bool | None]  # {"2d": ..., "5d": ..., "10d": ...}
    debate_substantive: bool | None
    notes: str
    judged_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Judgment:
        return cls(**d)


@dataclass
class Audit:
    """A retroactive check applied to an already-written Decision."""

    record_id: str
    audit_name: str
    result: str  # "pass" | "fail" | "flagged"
    detail: str
    audited_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Audit:
        return cls(**d)
