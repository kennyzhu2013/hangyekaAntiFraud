"""管线内部贯穿上下文。每个 service 只读上游字段、写自己的字段。"""

import re
from dataclasses import dataclass, field

from app.schemas.audit import AuditRequest, Turn

_WS = re.compile(r"\s+")


@dataclass
class PipelineContext:
    request: AuditRequest
    raw_turns: list[Turn] = field(default_factory=list)
    norm_turns: list[Turn] = field(default_factory=list)

    @property
    def agent_text(self) -> str:
        return "\n".join(t.text for t in self.norm_turns if t.speaker == "agent")

    def full_text_compact(self) -> str:
        """原文 + 归一文本去空白拼接，供证据子串校验用。"""
        raw = "".join(t.text for t in self.raw_turns)
        norm = "".join(t.text for t in self.norm_turns)
        return _WS.sub("", raw) + "\u0000" + _WS.sub("", norm)
