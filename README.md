# 重庆行业卡反诈质检策略 Agent（V1 MVP）

输入为通话转写文本的离线单条质检 API。采用 **规则引擎 + LLM 单 Agent** 架构：规则做确定性识别、证据抽取与风险初筛，LLM 基于规则命中与证据做最终归类、涉诈判定、说明生成与人工复核建议。

- 需求规范：`重庆行业卡质检复核规范V1.1_20260518.md`
- 方案文档：`思路.txt`（V1）、`思路V2_引入PyOD的质检策略Agent实施方案.md`（V2）、`实现思路分析_V2方案落地实现.md`
- 当前进度：V2 实施顺序的第 1-6 步（V1 MVP）。PyOD 异常检测通道（第 7-10 步）尚未接入，`ANOMALY_ENABLED` 保持 `false`。

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

LLM 走 OpenAI 兼容协议，通义/DeepSeek/Kimi 等均可通过 `LLM_BASE_URL` 接入。

## 请求示例

```bash
curl -X POST http://localhost:8000/api/v1/audit/transcript \
  -H "Content-Type: application/json" \
  -d '{"transcript": "left:您好，我是安逸花的人工客服，麻烦您添加一下我们下款经理的微信，您打开微信，点击最上方加号，在搜索框输入幺零九八，然后点添加到通讯录。right:好的。"}'
```

响应示例（字段说明见 `app/schemas/audit.py`）：

```json
{
  "is_violation": true,
  "is_fraud": false,
  "violation_type": "引流第三方平台",
  "risk_level": "高风险",
  "summary": "引流第三方平台：安逸花平台贷款推销，引导添加放款经理微信",
  "explanation": "……",
  "evidence": ["您打开微信，点击最上方加号"],
  "rule_hits": [{"rule_id": "CQ-DIVERT-WECHAT-USER-INITIATED", "...": "..."}],
  "needs_human_review": true,
  "review_reason": "none",
  "confidence": 0.92,
  "anomaly": {"enabled": false}
}
```

`review_reason` 枚举：`none` / `rule_hit_fraud`（涉诈规则命中）/ `rule_llm_conflict`（规则与模型冲突，已就高不就低）/ `low_confidence` / `anomaly_only`（V2 异常兜底）。

## 架构与目录

```
app/
  main.py                  FastAPI 入口（启动时加载规则库，fail-fast）
  api/audit.py             POST /api/v1/audit/transcript、GET /healthz
  schemas/audit.py         请求/响应/规则命中/LLM 裁决等 Pydantic 模型
  services/
    preprocess.py          说话人切分、噪声清理、同义词归一（raw/norm 双视图）
    rule_engine.py         YAML 规则编译、滑动窗口匹配、就高不就低合并
    evidence_extractor.py  从原文定位命中片段，输出带上下文的证据引用
    llm_agent.py           LLM 裁决（JSON mode + Pydantic + sanity check 三重保障）
    audit_pipeline.py      七步管线编排 + _fuse 融合后处理（硬约束代码强制执行）
  knowledge/
    chongqing_rules.yaml   规则库（规则 DSL：all/any/none + speaker + window）
    normalization.yaml     同义表达归一表
  prompts/
    chongqing_strategy_agent.md  系统提示词
    few_shots.json               典型样例
tests/
  fixtures/cases.json      黄金样本（20+ 条，覆盖规范典型场景）
  test_rule_engine.py      规则回归（每次改规则必须全绿）
  test_pipeline.py         融合矩阵硬约束测试
  test_api.py              接口冒烟
```

## 如何维护规则

1. 在 `app/knowledge/chongqing_rules.yaml` 中新增/修改规则条目（DSL 说明见文件头注释）。
2. **必须**同步在 `tests/fixtures/cases.json` 中补充正例（应命中）与反例（不应命中）。
3. 跑回归：`pytest tests/ -q`，全绿才允许合入。

风险等级判定口径以规范文档为唯一依据；涉诈类规则 `fraud_flag: true`，命中即强制高风险 + 人工复核（代码层兜底，不依赖 LLM）。

## 测试

```bash
pytest tests/ -q
```

测试不依赖外部 LLM（使用 Fake/Stub 客户端）。接入真实模型的效果评测建议另行使用黄金样本逐条比对。
