"""运行配置。环境变量优先，支持 .env 文件。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    # LLM（OpenAI 兼容协议）
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = "sk-placeholder"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.1

    # 融合策略
    low_conf_threshold: float = 0.6

    # 异常检测（V1 默认关闭，V2 接入 PyOD 后开启）
    anomaly_enabled: bool = False
    anomaly_model_dir: str = ""

    # 知识与提示词路径
    knowledge_dir: str = str(BASE_DIR / "knowledge")
    prompts_dir: str = str(BASE_DIR / "prompts")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
