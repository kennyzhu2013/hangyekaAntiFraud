"""管线与融合后处理测试（LLM 用 Fake 客户端，不出网）。

重点覆盖融合矩阵硬约束：
- 涉诈规则命中 → 强制高风险 + 人工复核；
- 规则与 LLM 冲突 → 就高不就低 + 送审；
- 异常分只能单向触发复核，不得修改类型/等级；
- LLM 输出失败 → 重试与回退。
"""

import asyncio
import json
from pathlib import Path

from app.config import Settings
from app.schemas.audit import AnomalyResult, AuditRequest
from app.services.audit_pipeline import AuditPipeline
from app.services.evidence_extractor import extract_evidence
from app.services.llm_agent import LLMAgent
from app.services.preprocess import Normalizer, preprocess
from app.services.rule_engine import RuleEngine

BASE = Path(__file__).resolve().parent
KNOWLEDGE = BASE.parent / "app" / "knowledge"

with open(BASE / "fixtures" / "cases.json", encoding="utf-8") as f:
    CASES = {c["name"]: c for c in json.load(f)}

NORMAL_VERDICT = {
    "is_violation": False,
    "is_fraud": False,
    "violation_type": None,
    "risk_level": "正常",
    "summary": "正常：未发现违规行为",
    "explanation": "通话内容未涉及违规或涉诈场景。",
    "evidence": [],
    "needs_human_review": False,
    "confidence": 0.9,
}


class FakeLLMClient:
    """按队列返回响应；队列耗尽后重复最后一条。"""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    async def complete_json(self, messages: list[dict]) -> str:
        self.calls += 1
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class StubAnomalyService:
    def __init__(self, result: AnomalyResult):
        self.result = result

    def score(self, ctx) -> AnomalyResult:
        return self.result


def build_pipeline(
    client: FakeLLMClient,
    anomaly_service=None,
    **settings_overrides,
) -> AuditPipeline:
    settings = Settings(
        llm_api_key="test",
        knowledge_dir=str(KNOWLEDGE),
        prompts_dir=str(BASE.parent / "app" / "prompts"),
        **settings_overrides,
    )
    normalizer = Normalizer.from_yaml(KNOWLEDGE / "normalization.yaml")
    engine = RuleEngine.from_yaml(KNOWLEDGE / "chongqing_rules.yaml")
    agent = LLMAgent(client, engine, settings)
    return AuditPipeline(normalizer, engine, agent, settings, anomaly_service)


def run(pipeline: AuditPipeline, transcript: str):
    return asyncio.run(pipeline.run(AuditRequest(transcript=transcript)))


def test_fraud_rule_hit_forces_high_risk_and_review():
    transcript = CASES["annual_report_fraud"]["transcript"]
    verdict = {
        **NORMAL_VERDICT,
        "is_violation": True,
        "is_fraud": True,
        "violation_type": "个体工商户年报补录收费",
        "risk_level": "高风险",
        "summary": "个体工商户年报补录收费：冒充税务引导小程序补录缴费，涉诈",
        "evidence": ["有可能产生一笔补录费用"],
        "needs_human_review": True,
        "confidence": 0.95,
    }
    pipeline = build_pipeline(FakeLLMClient([json.dumps(verdict, ensure_ascii=False)]))
    resp = run(pipeline, transcript)
    assert resp.is_fraud is True
    assert resp.risk_level == "高风险"
    assert resp.needs_human_review is True
    assert resp.review_reason == "rule_hit_fraud"
    assert any(h.rule_id == "CQ-FRAUD-ANNUAL-REPORT-001" for h in resp.rule_hits)


def test_fraud_rule_hit_overrides_llm_normal_verdict():
    """LLM 误判正常时，涉诈规则命中仍强制高风险（不依赖模型自觉）。"""
    transcript = CASES["annual_report_fraud"]["transcript"]
    pipeline = build_pipeline(
        FakeLLMClient([json.dumps(NORMAL_VERDICT, ensure_ascii=False)])
    )
    resp = run(pipeline, transcript)
    assert resp.is_fraud is True
    assert resp.is_violation is True
    assert resp.risk_level == "高风险"
    assert resp.review_reason == "rule_hit_fraud"
    assert resp.violation_type == "个体工商户年报补录收费"


def test_rule_llm_conflict_upgrades_risk():
    """规则候选高风险、LLM 判正常 → 就高不就低升级并送审。"""
    transcript = CASES["user_initiated_wechat_high_risk"]["transcript"]
    pipeline = build_pipeline(
        FakeLLMClient([json.dumps(NORMAL_VERDICT, ensure_ascii=False)])
    )
    resp = run(pipeline, transcript)
    assert resp.is_violation is True
    assert resp.risk_level == "高风险"
    assert resp.review_reason == "rule_llm_conflict"
    assert resp.violation_type == "引流第三方平台"
    assert resp.needs_human_review is True


def test_normal_call_passes_without_review():
    transcript = CASES["normal_call"]["transcript"]
    pipeline = build_pipeline(
        FakeLLMClient([json.dumps(NORMAL_VERDICT, ensure_ascii=False)])
    )
    resp = run(pipeline, transcript)
    assert resp.is_violation is False
    assert resp.risk_level == "正常"
    assert resp.needs_human_review is False
    assert resp.review_reason == "none"
    assert resp.anomaly.enabled is False


def test_llm_failure_falls_back_to_human_review():
    transcript = CASES["normal_call"]["transcript"]
    client = FakeLLMClient(["这不是JSON"])
    pipeline = build_pipeline(client)
    resp = run(pipeline, transcript)
    assert client.calls == 3
    assert resp.needs_human_review is True
    assert resp.review_reason == "low_confidence"
    assert resp.confidence == 0.0
    assert "无法判定" in resp.summary


def test_sanity_rejection_then_retry_succeeds():
    """第一次输出编造证据被拒，第二次合规输出被采纳。"""
    transcript = CASES["normal_call"]["transcript"]
    bad = {
        **NORMAL_VERDICT,
        "is_violation": True,
        "violation_type": "贷款相关",
        "risk_level": "低风险",
        "summary": "贷款相关：测试",
        "evidence": ["这句话完全不在原文里"],
    }
    client = FakeLLMClient(
        [
            json.dumps(bad, ensure_ascii=False),
            json.dumps(NORMAL_VERDICT, ensure_ascii=False),
        ]
    )
    pipeline = build_pipeline(client)
    resp = run(pipeline, transcript)
    assert client.calls == 2
    assert resp.is_violation is False


def test_anomaly_only_triggers_review_without_changing_verdict():
    """异常分只单向触发复核：结论保持正常，类型/等级不被修改。"""
    transcript = CASES["normal_call"]["transcript"]
    anomaly = AnomalyResult(
        enabled=True,
        score=0.91,
        is_outlier=True,
        top_deviating_features=["操作指令密度"],
        model_version="test-v1",
    )
    pipeline = build_pipeline(
        FakeLLMClient([json.dumps(NORMAL_VERDICT, ensure_ascii=False)]),
        anomaly_service=StubAnomalyService(anomaly),
        anomaly_enabled=True,
    )
    resp = run(pipeline, transcript)
    assert resp.is_violation is False
    assert resp.risk_level == "正常"
    assert resp.violation_type is None
    assert resp.needs_human_review is True
    assert resp.review_reason == "anomaly_only"
    assert resp.anomaly.score == 0.91


def test_anomaly_disabled_by_default():
    transcript = CASES["normal_call"]["transcript"]
    anomaly = AnomalyResult(enabled=True, score=0.99, is_outlier=True)
    pipeline = build_pipeline(
        FakeLLMClient([json.dumps(NORMAL_VERDICT, ensure_ascii=False)]),
        anomaly_service=StubAnomalyService(anomaly),
        # anomaly_enabled 默认 False：即使注入了服务也不生效
    )
    resp = run(pipeline, transcript)
    assert resp.anomaly.enabled is False
    assert resp.review_reason == "none"


def test_low_confidence_triggers_review():
    transcript = CASES["normal_call"]["transcript"]
    verdict = {**NORMAL_VERDICT, "confidence": 0.3}
    pipeline = build_pipeline(FakeLLMClient([json.dumps(verdict, ensure_ascii=False)]))
    resp = run(pipeline, transcript)
    assert resp.needs_human_review is True
    assert resp.review_reason == "low_confidence"


def test_evidence_extraction_returns_raw_fragments():
    case = CASES["annual_report_fraud"]
    settings = Settings(llm_api_key="test")
    normalizer = Normalizer.from_yaml(KNOWLEDGE / "normalization.yaml")
    engine = RuleEngine.from_yaml(KNOWLEDGE / "chongqing_rules.yaml")
    ctx = preprocess(AuditRequest(transcript=case["transcript"]), normalizer)
    hits = engine.evaluate(ctx.norm_turns)
    evidence = extract_evidence(ctx, hits)
    assert evidence
    full_raw = "".join(t.text for t in ctx.raw_turns)
    full_norm = "".join(t.text for t in ctx.norm_turns)
    for frag in evidence:
        body = frag.strip("…")
        assert body in full_raw or body in full_norm
