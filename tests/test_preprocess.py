from pathlib import Path

from app.schemas.audit import AuditRequest
from app.services.preprocess import Normalizer, parse_transcript, preprocess

KNOWLEDGE = Path(__file__).resolve().parent.parent / "app" / "knowledge"


def _normalizer() -> Normalizer:
    return Normalizer.from_yaml(KNOWLEDGE / "normalization.yaml")


def test_parse_left_right_alternating():
    turns = parse_transcript("left:你好 right:嗯，哪位 left:我是客服")
    assert [t.speaker for t in turns] == ["agent", "customer", "agent"]
    assert turns[0].text == "你好"
    assert turns[1].text == "嗯，哪位"
    assert [t.index for t in turns] == [0, 1, 2]


def test_parse_merges_consecutive_same_speaker():
    turns = parse_transcript("left:第一段left:第二段right:好的")
    assert len(turns) == 2
    assert turns[0].text == "第一段第二段"


def test_parse_inline_speaker_switch():
    # 规范案例中常见的句内说话人切换
    turns = parse_transcript("right:冬天left:今年是没有right:如果是这样")
    assert [t.speaker for t in turns] == ["customer", "agent", "customer"]


def test_parse_no_speaker_marker_defaults_to_agent():
    turns = parse_transcript("您好，请问是张先生吗")
    assert len(turns) == 1
    assert turns[0].speaker == "agent"


def test_normalization_applied_to_norm_view_only():
    request = AuditRequest(transcript="left:加一下我们的企微，关注公众号领取")
    ctx = preprocess(request, _normalizer())
    assert "企业微信" in ctx.norm_turns[0].text
    assert "服务号" in ctx.norm_turns[0].text
    # 原文视图保持不变
    assert "企微" in ctx.raw_turns[0].text
    assert "公众号" in ctx.raw_turns[0].text


def test_structured_conversation_input():
    request = AuditRequest(
        conversation=[
            {"speaker": "left", "text": "你好"},
            {"speaker": "right", "text": "哪位"},
        ]
    )
    ctx = preprocess(request, _normalizer())
    assert [t.speaker for t in ctx.raw_turns] == ["agent", "customer"]
