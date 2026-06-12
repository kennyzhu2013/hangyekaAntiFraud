"""规则引擎：YAML 规则编译、滑动窗口匹配、就高不就低合并。

Python 只实现匹配机制，全部业务知识在 chongqing_rules.yaml 中配置化维护。
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.schemas.audit import RISK_ORDER, RuleHit, RuleVerdict, Turn


@dataclass(frozen=True)
class Clause:
    speaker: str | None
    patterns: tuple[re.Pattern, ...]


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    category: str
    subtype: str
    risk_level: str
    fraud_flag: bool
    priority: int
    window: int  # <=0 表示整通通话
    decision_notes: str
    all_: tuple[Clause, ...] = field(default=())
    any_: tuple[Clause, ...] = field(default=())
    none_: tuple[Clause, ...] = field(default=())


def _compile_clause(raw: dict) -> Clause:
    return Clause(
        speaker=raw.get("speaker"),
        patterns=tuple(re.compile(p) for p in raw["any_pattern"]),
    )


def _compile_rule(raw: dict) -> CompiledRule:
    match = raw.get("match") or {}
    return CompiledRule(
        rule_id=raw["rule_id"],
        category=raw["category"],
        subtype=raw.get("subtype", ""),
        risk_level=raw["risk_level"],
        fraud_flag=bool(raw.get("fraud_flag", False)),
        priority=int(raw.get("priority", 0)),
        window=int(raw.get("window", 0) or 0),
        decision_notes=raw.get("decision_notes", ""),
        all_=tuple(_compile_clause(c) for c in match.get("all", [])),
        any_=tuple(_compile_clause(c) for c in match.get("any", [])),
        none_=tuple(_compile_clause(c) for c in match.get("none", [])),
    )


def _clause_first_hit(clause: Clause, window: list[Turn]) -> tuple[int, str] | None:
    """子句在窗口内的首个命中：返回 (轮次index, 触发片段)。"""
    for turn in window:
        if clause.speaker and turn.speaker != clause.speaker:
            continue
        for pattern in clause.patterns:
            m = pattern.search(turn.text)
            if m:
                return turn.index, m.group(0)
    return None


def _clause_matched(clause: Clause, window: list[Turn]) -> bool:
    return _clause_first_hit(clause, window) is not None


class RuleEngine:
    def __init__(self, rules: list[CompiledRule]):
        self.rules = rules
        self._by_category: dict[str, list[CompiledRule]] = {}
        for r in rules:
            self._by_category.setdefault(r.category, []).append(r)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleEngine":
        with open(path, encoding="utf-8") as f:
            raw_rules = yaml.safe_load(f) or []
        return cls([_compile_rule(r) for r in raw_rules])

    def rules_for_categories(self, categories: set[str]) -> list[CompiledRule]:
        out: list[CompiledRule] = []
        for cat in categories:
            out.extend(self._by_category.get(cat, []))
        return out

    def evaluate(self, turns: list[Turn]) -> list[RuleHit]:
        hits: list[RuleHit] = []
        for rule in self.rules:
            hit = self._evaluate_rule(rule, turns)
            if hit:
                hits.append(hit)
        return hits

    def _evaluate_rule(self, rule: CompiledRule, turns: list[Turn]) -> RuleHit | None:
        if not turns:
            return None
        if rule.window <= 0 or rule.window >= len(turns):
            starts: range = range(1)  # 整通通话只需一个窗口
            size = len(turns)
        else:
            starts = range(len(turns) - rule.window + 1)
            size = rule.window

        for start in starts:
            window = turns[start : start + size]
            matched: list[tuple[int, str]] = []

            ok = True
            for clause in rule.all_:
                h = _clause_first_hit(clause, window)
                if h is None:
                    ok = False
                    break
                matched.append(h)
            if not ok:
                continue

            if rule.any_:
                any_hit = None
                for clause in rule.any_:
                    any_hit = _clause_first_hit(clause, window)
                    if any_hit:
                        break
                if any_hit is None:
                    continue
                matched.append(any_hit)

            if any(_clause_matched(c, window) for c in rule.none_):
                continue

            # 去重并保持顺序
            seen: set[tuple[int, str]] = set()
            uniq = [m for m in matched if not (m in seen or seen.add(m))]
            return RuleHit(
                rule_id=rule.rule_id,
                category=rule.category,
                subtype=rule.subtype,
                risk_level=rule.risk_level,  # type: ignore[arg-type]
                fraud_flag=rule.fraud_flag,
                priority=rule.priority,
                matched_turns=[m[0] for m in uniq],
                matched_text=[m[1] for m in uniq],
                decision_notes=rule.decision_notes,
            )
        return None


def merge_hits(hits: list[RuleHit]) -> RuleVerdict:
    """就高不就低合并：风险等级优先，同级按 priority。合规命中保留供 LLM 参考。"""
    violations = [h for h in hits if h.risk_level != "合规"]
    if not violations:
        return RuleVerdict(candidate_risk="正常", hits=hits)
    top = max(violations, key=lambda h: (RISK_ORDER[h.risk_level], h.priority))
    return RuleVerdict(
        candidate_category=top.category,
        candidate_subtype=top.subtype,
        candidate_risk=top.risk_level,
        fraud_candidate=any(h.fraud_flag for h in violations),
        hits=hits,
    )
