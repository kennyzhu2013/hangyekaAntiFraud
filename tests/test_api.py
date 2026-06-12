"""API 层冒烟测试：启动加载与接口结构（LLM 用桩替换，不出网）。"""

import json

from fastapi.testclient import TestClient

from app.main import app

NORMAL_VERDICT = json.dumps(
    {
        "is_violation": False,
        "is_fraud": False,
        "violation_type": None,
        "risk_level": "正常",
        "summary": "正常：未发现违规行为",
        "explanation": "无违规内容。",
        "evidence": [],
        "needs_human_review": False,
        "confidence": 0.9,
    },
    ensure_ascii=False,
)


class StubClient:
    async def complete_json(self, messages):
        return NORMAL_VERDICT


def test_healthz_and_audit_endpoint():
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["rules"] > 0

        # 桩掉 LLM 客户端后调用质检接口
        app.state.pipeline.agent.client = StubClient()
        resp = client.post(
            "/api/v1/audit/transcript",
            json={"transcript": "left:您好，您的快递放驿站了。right:好的谢谢。"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_violation"] is False
        assert body["risk_level"] == "正常"
        assert body["anomaly"]["enabled"] is False


def test_audit_endpoint_validates_input():
    with TestClient(app) as client:
        resp = client.post("/api/v1/audit/transcript", json={})
        assert resp.status_code == 422
