"""输入标准化：说话人切分、噪声清理、同义词归一。

产出双视图：raw_turns（证据展示用，保留原文）与 norm_turns（规则匹配用，
经同义词归一），两者 index 严格对齐。
"""

import re
from pathlib import Path

import yaml

from app.schemas.audit import AuditRequest, Turn
from app.schemas.context import PipelineContext

# left:/right:/agent:/customer: 说话人标记（ASR 文本中常粘连在句中）
_SPEAKER_TOKEN = re.compile(r"(left|right|agent|customer)\s*[:：]", re.IGNORECASE)
_SPEAKER_MAP = {"left": "agent", "agent": "agent", "right": "customer", "customer": "customer"}
_WS = re.compile(r"\s+")


class Normalizer:
    """同义表达归一。基于 normalization.yaml（标准词 -> 变体列表）单次构建正则。"""

    def __init__(self, mapping: dict[str, list[str]] | None):
        pairs: list[tuple[str, str]] = []
        for standard, variants in (mapping or {}).items():
            for variant in variants or []:
                if variant and variant != standard:
                    pairs.append((variant, standard))
        # 长变体优先，避免短词抢占（如"工作微信"先于"微信"类变体）
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        self._lookup = {v: s for v, s in pairs}
        self._pattern = (
            re.compile("|".join(re.escape(v) for v, _ in pairs)) if pairs else None
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Normalizer":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f) or {})

    def normalize(self, text: str) -> str:
        if not self._pattern:
            return text
        return self._pattern.sub(lambda m: self._lookup[m.group(0)], text)


def _clean(text: str) -> str:
    text = text.replace("\\*", "*")
    return _WS.sub("", text).strip()


def parse_transcript(raw: str) -> list[Turn]:
    """按 left:/right: 标记切分说话人，合并连续同侧文本为一轮。"""
    parts = _SPEAKER_TOKEN.split(raw)
    segments: list[tuple[str, str]] = []
    if len(parts) == 1:
        # 无说话人标记：整段视为外呼侧（规范部分案例为外呼方独白）
        text = _clean(raw)
        return [Turn(speaker="agent", index=0, text=text)] if text else []
    # parts 形如 [前导, spk1, text1, spk2, text2, ...]
    for i in range(1, len(parts) - 1, 2):
        speaker = _SPEAKER_MAP[parts[i].lower()]
        text = _clean(parts[i + 1])
        if text:
            segments.append((speaker, text))
    turns: list[Turn] = []
    for speaker, text in segments:
        if turns and turns[-1].speaker == speaker:
            turns[-1] = Turn(
                speaker=speaker, index=turns[-1].index, text=turns[-1].text + text
            )
        else:
            turns.append(Turn(speaker=speaker, index=len(turns), text=text))
    return turns


def preprocess(request: AuditRequest, normalizer: Normalizer) -> PipelineContext:
    if request.conversation:
        raw_turns: list[Turn] = []
        for item in request.conversation:
            speaker = _SPEAKER_MAP.get(item.speaker.lower())
            if speaker is None:
                raise ValueError(f"无法识别的说话人标记: {item.speaker}")
            text = _clean(item.text)
            if not text:
                continue
            if raw_turns and raw_turns[-1].speaker == speaker:
                raw_turns[-1] = Turn(
                    speaker=speaker,
                    index=raw_turns[-1].index,
                    text=raw_turns[-1].text + text,
                )
            else:
                raw_turns.append(Turn(speaker=speaker, index=len(raw_turns), text=text))
    else:
        raw_turns = parse_transcript(request.transcript or "")

    norm_turns = [
        Turn(speaker=t.speaker, index=t.index, text=normalizer.normalize(t.text))
        for t in raw_turns
    ]
    return PipelineContext(request=request, raw_turns=raw_turns, norm_turns=norm_turns)
