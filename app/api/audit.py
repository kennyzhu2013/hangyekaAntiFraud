"""质检接口路由。"""

from fastapi import APIRouter, Request

from app.schemas.audit import AuditRequest, AuditResponse

router = APIRouter()


@router.post("/api/v1/audit/transcript", response_model=AuditResponse)
async def audit_transcript(payload: AuditRequest, request: Request) -> AuditResponse:
    pipeline = request.app.state.pipeline
    return await pipeline.run(payload)


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    engine = request.app.state.pipeline.rule_engine
    return {"status": "ok", "rules": len(engine.rules)}
