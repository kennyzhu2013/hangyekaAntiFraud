"""质检总管线：编排七步流程，并在 _fuse 中以代码强制执行全部业务硬约束。

硬约束（不依赖 LLM 自觉）：
1. 涉诈成立（规则或 LLM）→ 风险等级强制高风险 + 人工复核；
2. 规则候选风险高于 LLM 结论 → 就高不就低 + 冲突标记送人工复核；
3. 异常分只能单向触发人工复核，不得修改违规类型与风险等级。
"""

import asyncio
import logging
from typing import Protocol

from app.config import Settings
from app.schemas.audit import (
    RISK_ORDER,
    AnomalyResult,
    AuditRequest,
    AuditResponse,
    LLMVerdict,
    RuleVerdict,
)
from app.schemas.context import PipelineContext
from app.services.evidence_extractor import extract_evidence
from app.services.llm_agent import LLMAgent
from app.services.preprocess import Normalizer, preprocess
from app.services.rule_engine import RuleEngine, merge_hits

logger = logging.getLogger(__name__)

ANOMALY_CONF_BOOST = 0.05


class BaseAnomalyService(Protocol):
    """V2 接入 PyOD 后的在线打分接口；V1 不提供实现。"""

    def score(self, ctx: PipelineContext) -> AnomalyResult: ...


class AuditPipeline:
    def __init__(
        self,
        normalizer: Normalizer,
        rule_engine: RuleEngine,
        agent: LLMAgent,
        settings: Settings,
        anomaly_service: BaseAnomalyService | None = None,
    ):
        self.normalizer = normalizer
        self.rule_engine = rule_engine
        self.agent = agent
        self.settings = settings
        self.anomaly_service = anomaly_service

    async def run(self, request: AuditRequest) -> AuditResponse:
        # 1 输入标准化
        ctx = preprocess(request, self.normalizer)
        # 2 规则命中 + 4 异常打分（并行；CPU 任务放线程，避免阻塞事件循环）
        rule_task = asyncio.to_thread(self.rule_engine.evaluate, ctx.norm_turns)
        anomaly_task = asyncio.to_thread(self._score_anomaly, ctx)
        hits, anomaly = await asyncio.gather(rule_task, anomaly_task)
        # 3 证据提取
        evidence = extract_evidence(ctx, hits)
        # 5 就高不就低合并
        rule_verdict = merge_hits(hits)
        # 6 LLM 最终裁决
        llm_verdict = await self.agent.adjudicate(ctx, rule_verdict, evidence, anomaly)
        # 7 融合后处理
        return self._fuse(rule_verdict, llm_verdict, anomaly, evidence)

    def _score_anomaly(self, ctx: PipelineContext) -> AnomalyResult:
        if not self.settings.anomaly_enabled or self.anomaly_service is None:
            return AnomalyResult(enabled=False)
        try:
            return self.anomaly_service.score(ctx)
        except Exception:
            logger.exception("异常打分降级")
            return AnomalyResult(enabled=False)

    def _fuse(
        self,
        rule_v: RuleVerdict,
        llm_v: LLMVerdict,
        anomaly: AnomalyResult,
        extracted_evidence: list[str],
    ) -> AuditResponse:
        is_fraud = llm_v.is_fraud or rule_v.fraud_candidate
        is_violation = llm_v.is_violation or is_fraud
        risk = "高风险" if is_fraud else llm_v.risk_level

        review = llm_v.needs_human_review
        reason = "low_confidence" if review else "none"
        explanation = llm_v.explanation

        # 硬规则1：涉诈规则命中 → 强制高风险 + 人工复核
        if rule_v.fraud_candidate:
            review, reason = True, "rule_hit_fraud"
            if not llm_v.is_fraud:
                explanation += (
                    f"（系统提示：命中涉诈规则[{rule_v.candidate_category}]，"
                    "已按规范强制高风险并送人工复核）"
                )

        # 硬规则2：规则候选风险高于当前结论 → 就高不就低 + 冲突送审
        if RISK_ORDER.get(rule_v.candidate_risk, 0) > RISK_ORDER.get(risk, 0):
            risk = rule_v.candidate_risk
            is_violation = True
            review, reason = True, "rule_llm_conflict"
            explanation += (
                f"（系统提示：规则引擎候选[{rule_v.candidate_category}/"
                f"{rule_v.candidate_risk}]高于模型结论，按就高不就低升级并送人工复核）"
            )

        # 硬规则3：异常分只单向触发复核，不改类型/等级
        confidence = llm_v.confidence
        if anomaly.enabled and anomaly.is_outlier:
            if not is_violation and not review:
                review, reason = True, "anomaly_only"
            elif is_violation:
                confidence = min(1.0, confidence + ANOMALY_CONF_BOOST)

        if confidence < self.settings.low_conf_threshold and not review:
            review, reason = True, "low_confidence"

        violation_type = llm_v.violation_type
        if is_violation and not violation_type:
            violation_type = rule_v.candidate_category or "其他"
        if not is_violation:
            violation_type = None
            risk = "正常"

        evidence = llm_v.evidence or extracted_evidence
        return AuditResponse(
            is_violation=is_violation,
            is_fraud=is_fraud,
            violation_type=violation_type,
            risk_level=risk,  # type: ignore[arg-type]
            summary=llm_v.summary,
            explanation=explanation,
            evidence=evidence,
            rule_hits=rule_v.hits,
            needs_human_review=review,
            review_reason=reason,  # type: ignore[arg-type]
            confidence=confidence,
            anomaly=anomaly,
        )
