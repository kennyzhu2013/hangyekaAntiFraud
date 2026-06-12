"""质检 API 的输入输出 Schema 与管线内部数据结构。"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

# 规则风险等级（规则库口径，含"合规"）
RuleRisk = Literal["合规", "低风险", "中风险", "高风险"]
# 最终结论风险等级（对外口径，无违规即"正常"）
FinalRisk = Literal["正常", "低风险", "中风险", "高风险"]

# 风险等级排序，"就高不就低"的依据
RISK_ORDER: dict[str, int] = {
    "正常": 0,
    "合规": 0,
    "低风险": 1,
    "中风险": 2,
    "高风险": 3,
}


class Turn(BaseModel):
    """标准化后的一轮对话。speaker 统一为 agent（外呼侧）/ customer（用户侧）。"""

    speaker: Literal["agent", "customer"]
    index: int
    text: str


class RawTurnIn(BaseModel):
    """请求中可选的结构化对话轮次，speaker 允许 left/right/agent/customer。"""

    speaker: str
    text: str


class AuditRequest(BaseModel):
    transcript: Optional[str] = Field(
        default=None, description="原始转写文本，带 left:/right: 说话人标记"
    )
    conversation: Optional[list[RawTurnIn]] = Field(
        default=None, description="可选的结构化对话，与 transcript 二选一"
    )
    scene_hint: Optional[str] = Field(default=None, description="可选业务场景提示")

    @model_validator(mode="after")
    def _at_least_one_input(self) -> "AuditRequest":
        if not self.transcript and not self.conversation:
            raise ValueError("transcript 与 conversation 至少需要提供一个")
        return self


class RuleHit(BaseModel):
    rule_id: str
    category: str
    subtype: str
    risk_level: RuleRisk
    fraud_flag: bool = False
    priority: int = 0
    matched_turns: list[int] = Field(default_factory=list)
    matched_text: list[str] = Field(default_factory=list)
    decision_notes: str = ""


class RuleVerdict(BaseModel):
    """规则引擎"就高不就低"合并后的候选结论。"""

    candidate_category: Optional[str] = None
    candidate_subtype: Optional[str] = None
    candidate_risk: str = "正常"
    fraud_candidate: bool = False
    hits: list[RuleHit] = Field(default_factory=list)


class DetectorScore(BaseModel):
    name: str
    score: float


class AnomalyResult(BaseModel):
    """PyOD 异常检测结果。V1 阶段恒为 enabled=False，V2 在线接入后填充。"""

    enabled: bool = False
    score: Optional[float] = None
    is_outlier: Optional[bool] = None
    detectors: list[DetectorScore] = Field(default_factory=list)
    top_deviating_features: list[str] = Field(default_factory=list)
    model_version: Optional[str] = None


class LLMVerdict(BaseModel):
    """LLM 最终裁决器的结构化输出。"""

    is_violation: bool
    is_fraud: bool
    violation_type: Optional[str] = None
    risk_level: FinalRisk
    summary: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    confidence: float = 0.5

    @model_validator(mode="after")
    def _clamp_confidence(self) -> "LLMVerdict":
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self


ReviewReason = Literal[
    "none", "rule_hit_fraud", "rule_llm_conflict", "low_confidence", "anomaly_only"
]


class AuditResponse(BaseModel):
    is_violation: bool
    is_fraud: bool
    violation_type: Optional[str] = None
    risk_level: FinalRisk
    summary: str
    explanation: str
    evidence: list[str] = Field(default_factory=list)
    rule_hits: list[RuleHit] = Field(default_factory=list)
    needs_human_review: bool
    review_reason: ReviewReason = "none"
    confidence: float
    anomaly: AnomalyResult = Field(default_factory=AnomalyResult)
