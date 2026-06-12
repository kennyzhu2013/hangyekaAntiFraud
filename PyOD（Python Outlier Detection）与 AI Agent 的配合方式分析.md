**PyOD（Python Outlier Detection）与 AI Agent 的配合方式分析**

PyOD v3 专门设计了 **Agentic（智能体驱动）** 支持，让 AI Agent（如 Claude、自定义 LangGraph/CrewAI Agent 等）能通过**自然语言对话**，专业地完成异常检测（Outlier/Anomaly Detection）全流程，而不需要用户或 Agent 本身是异常检测专家。

这是 PyOD 从传统 ML 库向 **Agent-ready 库** 转型的核心特性之一。

### 1. 核心机制：`od-expert` Skill + ADEngine

- **od-expert Skill**：这是 PyOD 内置的“专家知识包”（约 1000 行专家内容），包含：
  - 主决策树（数据模态判断：表格、时间序列、图、文本、图像）。
  - Top-10 常见陷阱检查（Pitfalls），例如特征未归一化、高维数据用距离-based 模型等。
  - 11 个自适应升级触发器（Escalation Triggers），决定什么时候自主运行、什么时候暂停问用户（例如医疗/金融高风险场景）。
  - 基准驱动的检测器选择（基于 ADBench 等 benchmark）。
  - 知识库（RAG-like），自动加载对应模态的参考文档。

- **ADEngine**：PyOD 的智能编排引擎（orchestration core），提供结构化 Session API，让 Agent 能可靠地驱动流程。

Agent 通过调用这些接口，把自然语言意图转化为可靠的异常检测工作流。

### 2. 工作流程（How It Works）

当用户/Agent 提出异常检测需求时：

1. **意图触发** — `od-expert` Skill 根据关键词自动激活。
2. **数据画像 + 决策** — 走主决策树判断模态，加载对应知识。
3. **陷阱检查** — 逐一检查 Top-10 Pitfalls（如尺度差异过大）。
4. **触发器评估** — 检查 11 个 Escalation Triggers（高风险领域、标签可用性、检测器分歧等）。
5. **规划检测器** — `engine.plan()` 基于 benchmark 选 Top-N 检测器（例如 KNN + IForest + LOF）。
6. **并行执行** — `engine.run()` 运行多个检测器，生成 rank-normalized consensus（共识结果）。
7. **质量评估** — 计算 separation、agreement、stability 等指标，给出 overall quality verdict。
8. **后置检查 + 迭代** — 再走触发器，若有问题则暂停或调整；支持 `engine.iterate(state, feedback)` 处理用户反馈。
9. **生成报告** — `engine.report()` 输出 Markdown/JSON，包含假设、洞察、caveats、top anomalies 解释。

**示例场景**（糖尿病筛查数据集）：
- Agent 自动识别“medical screening” → 触发高风险 Trigger → 强制双检测器验证 + 报告 caveat。
- 检测到特征尺度差异 347 倍 → 自动标注 Pitfall 并提醒缩放。

### 3. Agent 如何集成 PyOD（三种主要路径）

| 集成方式 | 适用 Agent | 安装/使用方式 | 特点 |
|----------|-----------|---------------|------|
| **Skill 安装** | Claude Code / Claude Desktop / Codex | `pyod install skill` 或 `--project` | 最简单，自然语言驱动最强，Skill 自动加载 |
| **MCP Server** | MCP 兼容 Agent | `pip install pyod[mcp]` → `pyod mcp serve` | 提供结构化 Tools，Agent 可调用具体函数 |
| **直接 Python 调用** | LangGraph / CrewAI / AutoGen / 自定义 Agent | `from pyod.utils.ad_engine import ADEngine` | 最灵活，可嵌入多 Agent 系统中，作为专业 OD 子 Agent |

### 4. 关键 API（Agent 需要调用的）

- `engine = ADEngine()`
- `state = engine.investigate(X)` → 一键全流程
- `engine.plan(state)` / `run(state)` / `analyze(state)` → 分步控制
- `engine.iterate(state, feedback)` → 支持多轮对话迭代
- `engine.report(state, format='json')` → 结构化输出，便于下游处理

### 5. 与你之前场景（通话质检+反诈）的结合潜力

在**贷款营销/催收/推销通话文本**场景中，你可以把 PyOD 作为一个**专业异常检测子 Agent**：

- 把通话文本向量化（embedding）后 → 喂给 PyOD Agent 检测“异常对话模式”（如异常高压、异常索要信息、客户异常配合等）。
- 或者对数值化特征（通话时长、情绪分数、关键词频率等）直接做 outlier detection。
- Agent 可以结合 PyOD 的 `od-expert` 自动处理高风险金融场景的 caveats，并生成可解释报告。

**优势**：PyOD 帮 Agent 解决了“选哪个检测器、怎么评估质量、避什么坑”的问题，让整体质检+反诈 Agent 更可靠、专业。

需要我给你一个**结合 LangGraph 的示例代码**（把 PyOD ADEngine 包装成一个 Tool Node），还是针对通话文本的具体应用思路？