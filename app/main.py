"""FastAPI 应用入口。

启动时完成：规则库加载与正则编译、归一表加载、LLM 客户端构建。
规则库加载失败应直接启动失败（fail-fast）；异常检测缺失只降级不阻断。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.audit import router
from app.config import Settings
from app.services.audit_pipeline import AuditPipeline
from app.services.llm_agent import LLMAgent, OpenAICompatClient
from app.services.preprocess import Normalizer
from app.services.rule_engine import RuleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_pipeline(settings: Settings) -> AuditPipeline:
    knowledge = Path(settings.knowledge_dir)
    normalizer = Normalizer.from_yaml(knowledge / "normalization.yaml")
    rule_engine = RuleEngine.from_yaml(knowledge / "chongqing_rules.yaml")
    agent = LLMAgent(OpenAICompatClient(settings), rule_engine, settings)
    logger.info("规则库加载完成：%d 条规则", len(rule_engine.rules))
    return AuditPipeline(normalizer, rule_engine, agent, settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.settings = settings
    app.state.pipeline = build_pipeline(settings)
    yield


app = FastAPI(
    title="重庆行业卡反诈质检策略 Agent",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("未处理异常: %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": str(exc)},
    )
