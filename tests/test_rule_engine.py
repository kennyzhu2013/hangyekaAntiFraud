"""规则引擎黄金样本回归。

cases.json 中的每条样本断言：必须命中的规则、不得命中的规则、
就高不就低合并后的候选类别/风险/涉诈标记。
规则库每次改动必须保证本回归全绿。
"""

import json
from pathlib import Path

import pytest

from app.schemas.audit import AuditRequest
from app.services.preprocess import Normalizer, preprocess
from app.services.rule_engine import RuleEngine, merge_hits

BASE = Path(__file__).resolve().parent
KNOWLEDGE = BASE.parent / "app" / "knowledge"

NORMALIZER = Normalizer.from_yaml(KNOWLEDGE / "normalization.yaml")
ENGINE = RuleEngine.from_yaml(KNOWLEDGE / "chongqing_rules.yaml")

with open(BASE / "fixtures" / "cases.json", encoding="utf-8") as f:
    CASES = json.load(f)


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_case(case: dict):
    ctx = preprocess(AuditRequest(transcript=case["transcript"]), NORMALIZER)
    hits = ENGINE.evaluate(ctx.norm_turns)
    hit_ids = {h.rule_id for h in hits}

    for rule_id in case["expect_rules"]:
        assert rule_id in hit_ids, f"应命中 {rule_id}，实际命中 {sorted(hit_ids)}"
    for rule_id in case["expect_not_rules"]:
        assert rule_id not in hit_ids, f"不应命中 {rule_id}，实际命中 {sorted(hit_ids)}"

    verdict = merge_hits(hits)
    assert verdict.candidate_category == case["expect_candidate_category"], (
        f"候选类别应为 {case['expect_candidate_category']}，"
        f"实际 {verdict.candidate_category}（命中 {sorted(hit_ids)}）"
    )
    assert verdict.candidate_risk == case["expect_candidate_risk"]
    assert verdict.fraud_candidate == case["expect_fraud_candidate"]


def test_hits_carry_evidence_anchors():
    """每个命中必须带可回溯的轮次与触发片段。"""
    case = next(c for c in CASES if c["name"] == "annual_report_fraud")
    ctx = preprocess(AuditRequest(transcript=case["transcript"]), NORMALIZER)
    hits = ENGINE.evaluate(ctx.norm_turns)
    assert hits
    for hit in hits:
        assert hit.matched_turns, hit.rule_id
        assert hit.matched_text, hit.rule_id
        for idx in hit.matched_turns:
            assert 0 <= idx < len(ctx.norm_turns)


def test_merge_prefers_higher_risk():
    """同时命中低风险与高风险规则时，就高不就低。"""
    case = next(c for c in CASES if c["name"] == "user_initiated_wechat_high_risk")
    ctx = preprocess(AuditRequest(transcript=case["transcript"]), NORMALIZER)
    verdict = merge_hits(ENGINE.evaluate(ctx.norm_turns))
    risks = {h.risk_level for h in verdict.hits}
    assert "低风险" in risks and "高风险" in risks
    assert verdict.candidate_risk == "高风险"
