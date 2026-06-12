"""证据提取：从原文定位规则命中片段，输出带上下文的引用。

证据必须与转写原文逐字一致（人工复核要求），因此优先在 raw_turns 中定位；
规则匹配发生在归一文本上，若归一改写导致原文找不到片段，回退用归一文本。
"""

from app.schemas.audit import RuleHit
from app.schemas.context import PipelineContext

CONTEXT_CHARS = 60
MAX_EVIDENCE = 8


def extract_evidence(ctx: PipelineContext, hits: list[RuleHit]) -> list[str]:
    seen: set[tuple[int, int]] = set()
    out: list[str] = []
    for hit in hits:
        if hit.risk_level == "合规":
            continue
        for turn_idx, snippet in zip(hit.matched_turns, hit.matched_text):
            if turn_idx >= len(ctx.raw_turns):
                continue
            raw = ctx.raw_turns[turn_idx].text
            pos = raw.find(snippet)
            base = raw
            if pos < 0:
                base = ctx.norm_turns[turn_idx].text
                pos = base.find(snippet)
            if pos < 0:
                pos, snippet = 0, base[:40]
            start = max(0, pos - CONTEXT_CHARS)
            end = min(len(base), pos + len(snippet) + CONTEXT_CHARS)
            key = (turn_idx, start // (CONTEXT_CHARS * 2))
            if key in seen:
                continue
            seen.add(key)
            fragment = (
                ("…" if start > 0 else "")
                + base[start:end]
                + ("…" if end < len(base) else "")
            )
            out.append(fragment)
            if len(out) >= MAX_EVIDENCE:
                return out
    return out
