"""LLM 最终裁决器。

结构化输出三重保障：JSON mode + Pydantic 校验 + 业务 sanity check，
失败带原因重试，最终回退"无法判定 + 强制人工复核"。
"""

import json
import logging
import re
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from app.config import Settings
from app.schemas.audit import AnomalyResult, LLMVerdict, RuleVerdict
from app.schemas.context import PipelineContext
from app.services.rule_engine import RuleEngine

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
MAX_TURN_CHARS = 600
MAX_TURNS = 80


class SanityError(Exception):
    """LLM 输出通过了 JSON 校验但违反业务硬约束。"""


class BaseLLMClient(Protocol):
    async def complete_json(self, messages: list[dict]) -> str: ...


class OpenAICompatClient:
    """OpenAI 兼容协议客户端（通义/DeepSeek/Kimi 等均可通过 base_url 接入）。"""

    def __init__(self, settings: Settings):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature

    async def complete_json(self, messages: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


def _fallback_verdict() -> LLMVerdict:
    return LLMVerdict(
        is_violation=False,
        is_fraud=False,
        violation_type=None,
        risk_level="正常",
        summary="无法判定：模型结构化输出失败，需人工复核",
        explanation="LLM 多次输出不符合结构化要求，系统回退为无法判定，请人工复核本通通话。",
        evidence=[],
        needs_human_review=True,
        confidence=0.0,
    )


class LLMAgent:
    def __init__(
        self,
        client: BaseLLMClient,
        rule_engine: RuleEngine,
        settings: Settings,
    ):
        self.client = client
        self.rule_engine = rule_engine
        self.settings = settings
        prompts = Path(settings.prompts_dir)
        self.system_prompt = (prompts / "chongqing_strategy_agent.md").read_text(
            encoding="utf-8"
        )
        self.few_shots: list[dict] = json.loads(
            (prompts / "few_shots.json").read_text(encoding="utf-8")
        )

    async def adjudicate(
        self,
        ctx: PipelineContext,
        rule_verdict: RuleVerdict,
        evidence: list[str],
        anomaly: AnomalyResult,
    ) -> LLMVerdict:
        messages = self._build_messages(ctx, rule_verdict, evidence, anomaly)
        for attempt in range(self.settings.llm_max_retries):
            raw = ""
            try:
                raw = await self.client.complete_json(messages)
                verdict = LLMVerdict.model_validate_json(raw)
                self._sanity_check(verdict, ctx)
                return verdict
            except (ValidationError, SanityError, json.JSONDecodeError) as exc:
                logger.warning("LLM 输出不合规（第 %d 次）: %s", attempt + 1, exc)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": f"上一次输出不符合要求：{exc}。请严格按系统提示词的 JSON 格式与字段约束重新输出。",
                    },
                ]
            except Exception:
                logger.exception("LLM 调用失败（第 %d 次）", attempt + 1)
        return _fallback_verdict()

    # ---------- 上下文组装 ----------

    def _build_messages(
        self,
        ctx: PipelineContext,
        rule_verdict: RuleVerdict,
        evidence: list[str],
        anomaly: AnomalyResult,
    ) -> list[dict]:
        sections: list[str] = []

        hit_categories = {h.category for h in rule_verdict.hits}
        hits_payload = [
            {
                "rule_id": h.rule_id,
                "category": h.category,
                "subtype": h.subtype,
                "risk_level": h.risk_level,
                "fraud_flag": h.fraud_flag,
                "matched_text": h.matched_text,
            }
            for h in rule_verdict.hits
        ]
        sections.append(
            "【规则引擎命中】\n"
            + (json.dumps(hits_payload, ensure_ascii=False, indent=1) if hits_payload else "无命中")
        )
        sections.append(
            "【规则候选结论（就高不就低合并，仅供参考）】\n"
            + json.dumps(
                {
                    "candidate_category": rule_verdict.candidate_category,
                    "candidate_subtype": rule_verdict.candidate_subtype,
                    "candidate_risk": rule_verdict.candidate_risk,
                    "fraud_candidate": rule_verdict.fraud_candidate,
                },
                ensure_ascii=False,
            )
        )

        # 命中类别的全部分支说明（含合规分支，呈现例外逻辑）
        branch_lines = [
            f"- [{r.risk_level}{'/涉诈' if r.fraud_flag else ''}] {r.category} / {r.subtype}：{r.decision_notes}"
            for r in self.rule_engine.rules_for_categories(hit_categories)
        ]
        if branch_lines:
            sections.append("【同类规则分支说明（注意其中的合规例外）】\n" + "\n".join(branch_lines))

        if evidence:
            sections.append("【证据片段】\n" + "\n".join(f"- {e}" for e in evidence))

        if anomaly.enabled and anomaly.score is not None:
            sections.append(
                f"【统计异常信号】异常分 {anomaly.score:.2f}"
                f"（{'超过' if anomaly.is_outlier else '未超过'}阈值），"
                f"偏离特征：{('、'.join(anomaly.top_deviating_features) or '无')}。"
                "注意：异常不等于违规，禁止据此判定类型或风险等级。"
            )

        shots = self._select_shots(hit_categories)
        if shots:
            shot_lines = [
                f"案例：{s['case']}\n输出：{json.dumps(s['output'], ensure_ascii=False)}"
                for s in shots
            ]
            sections.append("【参考样例】\n" + "\n\n".join(shot_lines))

        if ctx.request.scene_hint:
            sections.append(f"【业务场景提示】{ctx.request.scene_hint}")

        sections.append("【对话全文】\n" + self._render_conversation(ctx, rule_verdict))
        sections.append("请按系统提示词要求输出 JSON 结论。")

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n\n".join(sections)},
        ]

    def _select_shots(self, categories: set[str]) -> list[dict]:
        matched = [s for s in self.few_shots if s["category"] in categories]
        if not matched:
            matched = [s for s in self.few_shots if s["category"] == "正常"]
        return matched[:3]

    def _render_conversation(self, ctx: PipelineContext, rule_verdict: RuleVerdict) -> str:
        turns = ctx.norm_turns
        keep: set[int] | None = None
        if len(turns) > MAX_TURNS:
            keep = set(range(15)) | set(range(len(turns) - 10, len(turns)))
            for h in rule_verdict.hits:
                for idx in h.matched_turns:
                    keep.update(range(max(0, idx - 3), min(len(turns), idx + 4)))
        lines: list[str] = []
        skipped = False
        for t in turns:
            if keep is not None and t.index not in keep:
                if not skipped:
                    lines.append("……（中间无关轮次省略）……")
                    skipped = True
                continue
            skipped = False
            text = t.text if len(t.text) <= MAX_TURN_CHARS else t.text[:MAX_TURN_CHARS] + "…"
            role = "外呼" if t.speaker == "agent" else "用户"
            lines.append(f"[{t.index}] {role}: {text}")
        return "\n".join(lines)

    # ---------- 业务硬约束校验 ----------

    def _sanity_check(self, verdict: LLMVerdict, ctx: PipelineContext) -> None:
        full = ctx.full_text_compact()
        for ev in verdict.evidence:
            compact = _WS.sub("", ev).strip("…")
            if compact and compact not in full:
                raise SanityError(f"evidence 片段不是对话原文摘录: {ev[:50]}")
        if verdict.is_fraud:
            if not verdict.is_violation:
                raise SanityError("is_fraud=true 时 is_violation 必须为 true")
            if verdict.risk_level != "高风险":
                raise SanityError("涉诈成立时 risk_level 必须为高风险")
        if verdict.is_violation:
            if not verdict.violation_type:
                raise SanityError("is_violation=true 时必须给出 violation_type")
            if not verdict.evidence:
                raise SanityError("is_violation=true 时 evidence 不得为空")
            if not re.search(r"^[^：:]{2,30}[：:].+", verdict.summary):
                raise SanityError('summary 必须遵循"违规/涉诈类型：简要分析"格式')
            if verdict.risk_level == "正常":
                raise SanityError("is_violation=true 时 risk_level 不能为正常")
        else:
            if verdict.risk_level != "正常":
                raise SanityError("is_violation=false 时 risk_level 必须为正常")
