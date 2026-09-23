"""Append-only, per-ticker storage for the eval module's four event logs.

Nothing written here is ever rewritten: a correction is a new row (an Audit),
not an edit to an existing record. See docs/eval-framework-design.md §2 on
why an earlier draft's mutable, in-place-updated record was replaced with
this — settlements and judgments arrive at different times for the same
decision, and mutating a written record needed atomic temp-file-replace
writes and made auditing harder than just appending.

Storage is split one directory per ticker because every reader and writer
here already operates per-ticker (settlement looks up one ticker's prices at
a time), so this needs no cross-ticker index and no locking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tradingagents.eval.schema import Audit, Decision, Judgment, Settlement


@dataclass
class JoinedRecord:
    """One decision, with every settlement/judgment/audit recorded against it."""

    decision: Decision
    settlements: list[Settlement] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    audits: list[Audit] = field(default_factory=list)


class EvalStore:
    """Append-only JSONL store, one directory per ticker under ``root``."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def _ticker_dir(self, ticker: str) -> Path:
        d = self.root / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _append(self, path: Path, row: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _load(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    # --- decisions: hash-chained per ticker ---

    def append_decision(self, decision: Decision) -> Decision:
        """Fill in the hash chain against this ticker's prior decision, then
        append. Mutates and returns ``decision`` as actually written."""
        prior = self.load_decisions(decision.ticker)
        decision.prev_hash = prior[-1].record_hash if prior else ""
        decision.record_hash = decision.content_hash()
        self._append(self._ticker_dir(decision.ticker) / "decisions.jsonl", decision.to_dict())
        return decision

    def load_decisions(self, ticker: str) -> list[Decision]:
        return [Decision.from_dict(d) for d in self._load(self._ticker_dir(ticker) / "decisions.jsonl")]

    def verify_chain(self, ticker: str) -> list[str]:
        """Record ids whose stored hash doesn't match its recomputed content
        hash, or whose prev_hash doesn't match the prior record's hash —
        empty when the chain is intact."""
        broken = []
        prev_hash = ""
        for d in self.load_decisions(ticker):
            if d.prev_hash != prev_hash or d.record_hash != d.content_hash():
                broken.append(d.record_id)
            prev_hash = d.record_hash
        return broken

    # --- settlements / judgments / audits: flat append, no chain ---

    def append_settlement(self, ticker: str, settlement: Settlement) -> None:
        self._append(self._ticker_dir(ticker) / "settlements.jsonl", settlement.to_dict())

    def load_settlements(self, ticker: str) -> list[Settlement]:
        return [Settlement.from_dict(d) for d in self._load(self._ticker_dir(ticker) / "settlements.jsonl")]

    def append_judgment(self, ticker: str, judgment: Judgment) -> None:
        self._append(self._ticker_dir(ticker) / "judgments.jsonl", judgment.to_dict())

    def load_judgments(self, ticker: str) -> list[Judgment]:
        return [Judgment.from_dict(d) for d in self._load(self._ticker_dir(ticker) / "judgments.jsonl")]

    def append_audit(self, ticker: str, audit: Audit) -> None:
        self._append(self._ticker_dir(ticker) / "audits.jsonl", audit.to_dict())

    def load_audits(self, ticker: str) -> list[Audit]:
        return [Audit.from_dict(d) for d in self._load(self._ticker_dir(ticker) / "audits.jsonl")]

    # --- read-time join ---

    def join_records(self, ticker: str) -> list[JoinedRecord]:
        """Assemble each decision with the settlements/judgments/audits
        recorded against it. The join happens here, at read time — nothing
        is stored redundantly across the four logs."""
        joined = {d.record_id: JoinedRecord(decision=d) for d in self.load_decisions(ticker)}
        for s in self.load_settlements(ticker):
            if s.record_id in joined:
                joined[s.record_id].settlements.append(s)
        for j in self.load_judgments(ticker):
            if j.record_id in joined:
                joined[j.record_id].judgments.append(j)
        for a in self.load_audits(ticker):
            if a.record_id in joined:
                joined[a.record_id].audits.append(a)
        return list(joined.values())
