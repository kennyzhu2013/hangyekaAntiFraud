# 重庆行业卡反诈质检策略 Agent 实施方案 V2（引入 PyOD）

> 本方案在 `思路.txt`（V1：规则引擎 + LLM 单 Agent + FastAPI 离线单条 API）的基础上，分析引入 **PyOD v3（Python Outlier Detection，含 ADEngine 智能编排引擎与 od-expert Skill）** 后实施方案需要做的修改，并给出修订后的完整实施计划。
>
> 参考输入：
> - 需求规范：`重庆行业卡质检复核规范V1.1_20260518.md`
> - 原实施计划：`思路.txt`
> - PyOD 能力分析：`PyOD（Python Outlier Detection）与 AI Agent 的配合方式分析.md`

---

## 一、结论摘要（TL;DR）

1. **主判定链路不变**：质检结论必须输出“正常/违规、风险等级、判定说明（违规/涉诈类型 + 分析）”，这是**规则语义判断**，PyOD 的统计异常检测**不能替代**规则引擎与 LLM 裁决器，规范中“就高不就低”“涉诈一律高风险”等逻辑仍由规则 + LLM 承担。
2. **PyOD 以“第三信号通道 + 离线发现工具”身份接入**，承担两类 V1 没有覆盖的能力：
   - **在线**：对单条通话计算**异常分（anomaly score）**，作为置信度校准与人工复核优先级的辅助信号，重点兜底“规则未命中但模式可疑”的盲区通话。
   - **离线**：用 PyOD ADEngine 对每日通话批量做无监督异常筛查，**发现规则库尚未覆盖的新型话术/新型诈骗套路**，反哺规则迭代与 Few-shot 样本沉淀（对应规范“违规场景涉诈判断与补充”一章中“新增类型违规场景”的诉求）。
3. **架构由“单管线”升级为“双通道”**：
   - 在线通道：FastAPI 单条 API，管线中插入“特征提取 → 预训练 PyOD 检测器打分”一步；
   - 离线通道：批量训练/筛查作业，使用 ADEngine 的 `investigate / plan / run / analyze / report` 全流程。
4. **集成路径选“直接 Python 调用”**（`from pyod.utils.ad_engine import ADEngine`），不走 Claude Skill、不强依赖 MCP Server——因为本系统是自定义 FastAPI Agent，需要把 PyOD 嵌入自己的管线内，这是 PyOD 三种集成方式中最灵活的一种。
5. **必须新增的工程内容**：特征工程层（文本向量化 + 行为统计特征）、异常检测服务、离线训练与阈值标定脚本、模型工件管理、Schema 新增异常字段、提示词新增“异常信号使用纪律”。
6. **必须坚守的边界**：**异常 ≠ 违规**。异常分只能影响 `needs_human_review`、`confidence` 与复核优先级，**不允许**单凭异常分输出违规类型或提升风险等级——风险等级判定口径仍以规范为唯一依据。

---

## 二、PyOD 与本需求的匹配度分析

### 2.1 PyOD 能补什么

| V1 方案的短板 | PyOD 的对应能力 |
|---|---|
| 规则只能识别“已知套路”，规范也明确会出现**新增类型违规场景**，V1 对未知话术只能回退“人工复核”，没有主动发现机制 | 无监督异常检测天然适合“发现不知道自己不知道的东西”：对历史通话特征分布建模，新型话术会表现为统计离群点 |
| LLM 置信度（`confidence`）缺乏独立校准来源，规则未命中时模型容易“想当然” | rank-normalized consensus（多检测器共识分）提供与规则、LLM 都独立的第三方信号，可用于校准置信度与触发人工复核 |
| 每日 AI 质检异常通话量大，人工复核没有优先级排序依据 | 异常分可直接用于复核队列排序：分数越高越优先 |
| 检测器选型、参数、质量评估需要异常检测专业知识，团队没有 OD 专家 | od-expert Skill + ADEngine 内置：主决策树（模态判断）、Top-10 Pitfalls 检查（如特征未归一化、高维数据误用距离类模型）、基于 ADBench benchmark 的检测器选择、separation/agreement/stability 质量评估 |
| 金融/反诈是高风险领域，自动化结论需要保守策略 | ADEngine 的 11 个 Escalation Triggers 会在高风险领域（金融即属此类）强制双检测器验证、输出 caveat、必要时暂停等待人工确认 |

### 2.2 PyOD 不能做什么（边界，决定了方案怎么改）

1. **PyOD 不理解业务语义。** 它无法区分“外呼人员主动加用户微信（合规）”与“引导用户主动加微信（高风险）”——这种依赖“主动方 + 添加对象 + 业务场景”的细粒度分支判断，仍然只能靠规则引擎 + LLM。规范中全部风险等级口径（高/中/低风险、涉诈固定高风险、就高不就低）与 PyOD 无关。
2. **异常是相对群体分布而言的。** 如果某类违规话术（如安逸花类网贷推销）在通话总体中大量出现，它在统计上反而“不异常”；反之一通完全正常但话题罕见的通话可能得高异常分。因此异常分**只能作为辅助信号**，不能作为违规判定依据。
3. **PyOD 的 `fit` 需要样本集合，单条样本无法独立建模。** 在线单条 API 场景下，必须**先离线在历史语料上训练好检测器**，在线只调用 `decision_function` 打分。ADEngine 的 `investigate` 全流程（画像→规划→并行执行→质量评估→报告）本质是批处理工作流，应放在离线通道，而不是塞进单条请求的同步链路里。
4. **可解释性形式不同。** 规范要求判定说明为“违规/涉诈类型 + 简要分析”，并能回溯到原文证据；PyOD 的解释是“哪些特征维度偏离、偏离多少”。两种解释不能混用：异常解释只进入人工复核的参考信息，不进入对外的判定说明正文。

### 2.3 集成路径选择

PyOD 提供三种 Agent 集成方式，本项目选择如下：

| 集成方式 | 是否采用 | 理由 |
|---|---|---|
| Skill 安装（Claude Code / Desktop） | 否 | 本系统是自部署 FastAPI 服务，不在 Claude 桌面环境内运行 |
| MCP Server（`pyod mcp serve`） | 本期否，预留 | 若后续把质检 Agent 改造成多 Agent 编排（LangGraph 等），可把 PyOD MCP 作为独立工具节点接入；首期单体服务无此必要 |
| **直接 Python 调用 ADEngine** | **是** | 与 FastAPI 同进程/同代码库，离线训练与在线打分都可控，延迟与部署最简单 |

---

## 三、修订后的总体架构：双通道

```
┌─────────────────────────────  在线通道（单条 API，同步）  ─────────────────────────────┐
│                                                                                        │
│  请求 → 预处理 → 规则引擎 → 证据提取 ─┬→ LLM 最终裁决（规则命中 + 证据 + 异常信号）→ 响应 │
│                  │                    │                                                │
│                  └→ 特征提取 → 预训练 PyOD 检测器打分（anomaly score）─┘                 │
│                       （加载离线产出的模型工件，仅 decision_function，无 fit）            │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────  离线通道（批量作业，异步）  ─────────────────────────────┐
│                                                                                        │
│  历史/每日通话语料 → 特征提取 → ADEngine.investigate / plan / run / analyze            │
│      → 多检测器共识 + 质量评估（separation / agreement / stability）                    │
│      → engine.report(format='json'/'markdown')                                         │
│      → 产出：①训练好的检测器工件 + 标定阈值（供在线加载）                                │
│             ②高异常分通话清单（人工复核优先队列 / 新型话术候选）                         │
│             ③复盘报告（假设、洞察、caveats、top anomalies 解释）                        │
│      → 人工确认新型套路后 → 回写 chongqing_rules.yaml 与 few_shots.json（规则迭代闭环）  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

要点：

- **在线通道对 V1 是“插桩式”修改**：在规则引擎与 LLM 之间并联一步异常打分，不改变原有规则优先、LLM 裁决的骨架；PyOD 打分失败或模型工件缺失时**自动降级**为 V1 行为（异常字段置空），不影响主链路可用性。
- **离线通道是全新增量**：承担训练、阈值标定、批量筛查与规则迭代闭环，是 PyOD 价值最大的部分。

---

## 四、特征工程设计（新增，PyOD 的输入）

PyOD 消费的是数值特征矩阵，必须新增一层把对话文本转成特征向量。两路特征拼接：

### 4.1 语义向量特征

- 对整通对话（或仅外呼侧 `left` 文本）做 embedding（本地 sentence-transformers 类模型或与 LLM 同源的 embedding API）。
- 高维 embedding 直接喂距离类检测器（KNN/LOF）是 od-expert 明确列出的 Pitfall，处理方式二选一：
  - 降维（PCA/UMAP）到几十维后再进检测器；
  - 或选用对高维稳健的检测器（IForest、ECOD、COPOD）。
- ADEngine 的陷阱检查会自动提示该问题，离线训练时以其建议为准。

### 4.2 行为/统计特征（业务侧手工特征，可解释性强）

围绕规范中的违规要素设计，示例：

| 特征组 | 示例特征 | 对应规范关注点 |
|---|---|---|
| 通话结构 | 通话时长、轮次数、外呼方/用户方字数比、用户平均应答长度 | 单方面长篇灌输（如保健品洗脑、商业地产话术）的结构特征 |
| 引导操作密度 | “打开微信/点击加号/添加朋友/搜索/发送/置顶/取消免打扰”等操作指令出现次数与连续度 | 引流第三方平台、证券投资类“分步遥控操作”的套路 |
| 资金动作信号 | 定金/保证金/解冻金/补录费/邮费/会费等资金词频，金额数字出现次数 | 提前收费、法律服务收费、年报补录收费 |
| 平台引流信号 | 服务号/公众号/企业微信/小程序/APP/链接/二维码/屏幕共享 词频 | 非个人微信引流（高风险）要素 |
| 敏感信息信号 | 身份证号/住址/验证码/征信 等敏感词频 | 其他敏感信息（高风险）、违规催收 |
| 规则引擎输出 | 命中规则数、命中最高风险等级（序数化）、命中类别 one-hot | 让异常检测“知道”规则视角，离群点更聚焦于规则盲区 |

> 设计原则：手工特征必须由 `normalization.yaml`（V1 已规划的同义词归一层）先归一再统计，避免“企微/工作微信”逃逸。特征定义集中放在配置文件中，与规则库一起做版本管理。

### 4.3 特征规范化

- 所有手工特征做标准化/分位数缩放后再与降维向量拼接（PyOD 文档示例中“特征尺度差异 347 倍”正是 Top-10 Pitfall，ADEngine 会标注，但在线打分链路必须固化与离线训练**完全一致**的缩放器，随模型工件一起序列化）。

---

## 五、决策融合策略：规则 × LLM × 异常分

异常分进入决策的方式严格受限，融合矩阵如下：

| 规则命中情况 | 异常分 | 系统行为 |
|---|---|---|
| 命中明确违规/涉诈规则 | 高 | 结论照常由规则 + LLM 给出；`confidence` 上调；复核优先级提升 |
| 命中明确违规/涉诈规则 | 低 | 结论照常给出；`confidence` 不因低异常分下调（已知套路大量出现时统计上不异常，属预期） |
| 未命中规则、LLM 判正常 | 高 | **关键兜底场景**：结论可保持“正常”，但 `needs_human_review=true`，`review_reason="anomaly_only"`，进入人工复核队列高优先级——这是 PyOD 对 V1 的最大增量 |
| 未命中规则、LLM 判正常 | 低 | 正常放行，`needs_human_review=false`（除非 LLM 自身置信不足） |
| 规则与 LLM 结论冲突 | 任意 | 维持 V1 策略：强制人工复核；异常分仅作为复核排序参考 |

硬约束（写入代码与提示词双重保障）：

1. 异常分**不得**生成或修改 `violation_type` / `fraud_tags`；
2. 异常分**不得**提升 `risk_level`（风险等级只由规范口径决定）；
3. 异常分只能单向影响：`needs_human_review`（只能从 false 变 true，不能反向）、`confidence`（小幅校准）、复核优先级排序。

---

## 六、对 V1 方案的逐项修改清单

### 6.1 目录与文件（在 V1 规划基础上的增删改）

**新增：**

- `app/services/feature_extractor.py`
  - 作用：实现第四节的特征工程（embedding 调用 + 行为统计特征 + 缩放），在线/离线共用同一份实现，保证训练-推理一致性。
- `app/services/anomaly_service.py`
  - 作用：封装 PyOD。
  - 在线侧：加载 `models/` 下的检测器工件与缩放器，提供 `score(features) -> AnomalyResult`（含共识分、各检测器分、是否超阈值、top 偏离特征）；模型缺失/异常时返回空结果并打日志，不阻断主链路。
  - 离线侧：封装 `ADEngine` 的 `investigate / plan / run / analyze / iterate / report` 调用。
- `app/knowledge/feature_spec.yaml`
  - 作用：行为特征的词表与统计口径配置（复用 `normalization.yaml` 的归一结果）。
- `scripts/train_anomaly.py`
  - 作用：离线训练入口。读取历史语料 → 特征提取 → `ADEngine.investigate(X)`（或 `plan`+`run` 分步控制）→ 质量评估通过后导出检测器工件、缩放器、阈值、训练报告。
- `scripts/batch_screen.py`
  - 作用：每日批量筛查作业。对当日通话打分 → 产出高异常分清单（JSON/Markdown，用 `engine.report` 生成）→ 推送人工复核队列。
- `models/`（工件目录，含版本号与训练元数据 manifest）
- `tests/test_anomaly_service.py`

**修改：**

- `app/schemas/audit.py`：响应模型新增字段（见 6.2）。
- `app/services/audit_pipeline.py`：固定流程由 6 步改为 7 步（见 6.3）。
- `app/services/llm_agent.py`：上下文拼装新增异常信号段；输出解析新增对“异常仅作复核建议”约束的校验。
- `app/prompts/chongqing_strategy_agent.md`：新增异常信号使用纪律条款（见 6.4）。
- `requirements.txt`：新增 `pyod`（v3+）、`numpy`、`scikit-learn`、`joblib`、embedding 相关依赖（如 `sentence-transformers`，或仅 HTTP 客户端走 embedding API）。
- `.env.example`：新增 `ANOMALY_MODEL_DIR`、`ANOMALY_SCORE_THRESHOLD`、`EMBEDDING_*` 配置；新增开关 `ANOMALY_ENABLED`（默认可关，便于首期灰度）。
- `README.md`：补充离线训练、批量筛查、模型工件更新流程说明。

**不变：**

- `app/main.py`、`app/api/audit.py`、`app/services/preprocess.py`、`app/services/rule_engine.py`、`app/services/evidence_extractor.py`、`app/knowledge/chongqing_rules.yaml`、`app/knowledge/normalization.yaml`、`app/prompts/few_shots.json` 的 V1 职责全部保留。

### 6.2 输出契约（Output Contract）变更

在 V1 响应结构上**追加**（不破坏既有字段）：

```json
{
  "is_violation": true,
  "is_fraud": true,
  "violation_type": "个体工商户年报补录收费",
  "risk_level": "高风险",
  "summary": "个体工商户年报补录收费：冒充税务服务主体，引导在小程序补录缴费，属于涉诈。",
  "explanation": "通话中自称税务服务中心，要求用户在小程序完成补录并支付费用。规范明确指出年报申报本身免费，补录收费属于诈骗套路。",
  "evidence": ["...原文片段..."],
  "rule_hits": [{"rule_id": "CQ-FRAUD-ANNUAL-REPORT-001", "category": "个体工商户年报补录收费", "risk_level": "高风险"}],
  "needs_human_review": true,
  "review_reason": "rule_hit_fraud",
  "confidence": 0.93,

  "anomaly": {
    "enabled": true,
    "score": 0.87,
    "is_outlier": true,
    "detectors": [
      {"name": "IForest", "score": 0.83},
      {"name": "ECOD", "score": 0.91},
      {"name": "KNN", "score": 0.86}
    ],
    "consensus": "rank_normalized",
    "top_deviating_features": ["补录费词频", "操作指令密度", "验证码词频"],
    "model_version": "anomaly-2026Q2-v1"
  }
}
```

说明：

- `anomaly` 整块可空（`enabled=false` 或打分失败时），调用方不依赖其存在；
- `review_reason` 为新增枚举：`rule_hit_fraud` / `rule_llm_conflict` / `low_confidence` / `anomaly_only` 等，便于复核队列分流与后续统计 PyOD 的实际拦截贡献。

### 6.3 在线质检管线（`audit_pipeline.py`）固定流程修订

V1 六步 → V2 七步：

1. 输入标准化（不变）
2. 规则命中（不变）
3. 证据提取（不变）
4. **特征提取 + 预训练 PyOD 检测器打分（新增，与 3 可并行执行；失败则降级跳过）**
5. 规则优先级合并（不变）
6. 调用 LLM 做最终裁决与说明生成（输入新增异常信号摘要）
7. 输出标准化结果（新增 anomaly 块；按第五节融合矩阵后处理 `needs_human_review` / `confidence`——注意该后处理在代码中执行，不依赖 LLM 自觉）

### 6.4 提示词修订（`chongqing_strategy_agent.md` 新增条款）

在 V1 提示词要求之上追加：

- 输入中可能包含“统计异常信号”（异常分、偏离特征）；它表示该通话在历史通话分布中的离群程度，**不代表违规或涉诈**。
- **禁止**以异常分作为判定违规类型、涉诈、风险等级的依据；类型与等级只能依据规则命中与原文证据。
- 当规则未命中、证据不足以判违规、但异常分高时：结论给“正常”或“无法判定”，同时建议人工复核，并在复核建议中引用偏离特征作为提示线索（不写入对外判定说明正文）。

### 6.5 离线训练与阈值标定流程（新增）

1. **语料准备**：收集历史通话转写（建议 ≥ 数千条，覆盖正常与各类违规），经 `preprocess.py` 标准化。
2. **特征提取**：`feature_extractor.py` 批量产出特征矩阵 X。
3. **ADEngine 训练**：
   - `state = engine.investigate(X)` 一键全流程；或 `engine.plan(state)`（基于 benchmark 选 Top-N 检测器，预期为 IForest/ECOD/KNN 一类组合）→ `engine.run(state)`（并行执行 + rank-normalized consensus）→ `engine.analyze(state)` 分步控制。
   - 关注 ADEngine 的 Pitfall 标注（尺度、维度问题）与 Escalation Triggers（金融高风险场景会触发强制双检测器验证与 caveat，按其要求执行）。
4. **质量评估**：separation / agreement / stability 指标 + overall quality verdict 达标才允许发布工件；不达标时用 `engine.iterate(state, feedback)` 携带反馈迭代。
5. **阈值标定**：用一批已有人工复核结论的通话回放打分，选定使“高异常分 → 确实值得人工看”的查准率可接受的阈值（建议以人工复核处理容量倒推 top-k% 分位数作为初始阈值）。
6. **工件发布**：序列化检测器 + 缩放器 + 阈值 + 训练报告（`engine.report(state, format='json')`）至 `models/`，带版本号；在线服务热加载或重启加载。
7. **周期重训**：按月或语料漂移监控触发，防止话术分布漂移导致异常分失效。

### 6.6 测试与回归修订

在 V1 测试规划上新增：

- `tests/test_anomaly_service.py`：
  - 特征提取的确定性（同输入同特征）；
  - 训练-推理缩放一致性；
  - 模型工件缺失/损坏时的优雅降级（管线不报错、anomaly 块为空）；
  - 融合矩阵后处理逻辑（异常分只能单向置位 `needs_human_review`，不得改 `risk_level` / `violation_type`）。
- 回归集（V1 的 10-20 条黄金样本）增加断言维度：引入 PyOD 后，黄金样本的 `is_violation` / `risk_level` / `violation_type` 输出必须与 V1 完全一致（证明 PyOD 未污染主判定链路）。
- 新增 2-3 条“规则未命中的杜撰新型话术”样本，验证 `anomaly_only` 复核兜底路径可触发。

---

## 七、修订后的实施步骤（Implementation Plan V2）

V1 的 Step 1-7 保持不变并优先完成（PyOD 不阻塞 MVP），随后追加：

- **Step 8. 特征工程层**
  - 实现 `feature_extractor.py` 与 `feature_spec.yaml`，先覆盖第四节的行为特征 + embedding。
  - 验证标准：对规范中的典型案例（年报补录、证券投资、保健品推销、正常通话）产出可解释、可复现的特征向量。
- **Step 9. 离线训练与批量筛查（PyOD 价值首先在这里兑现）**
  - 实现 `scripts/train_anomaly.py` 与 `scripts/batch_screen.py`，跑通 ADEngine 全流程并产出首版模型工件与筛查报告。
  - 验证标准：质量评估指标达标；批量筛查报告中 top anomalies 经人工抽检“值得一看”的比例可接受（建议首版目标 ≥ 50%）。
- **Step 10. 在线异常打分接入**
  - 实现 `anomaly_service.py` 在线侧，管线插桩（6.3），Schema 扩展（6.2），融合后处理（第五节），`ANOMALY_ENABLED` 开关灰度。
  - 验证标准：黄金样本主判定结果与 V1 完全一致；杜撰新型话术样本可触发 `anomaly_only` 复核；模型缺失时服务行为等同 V1。
- **Step 11. 提示词与复核闭环**
  - 提示词新增异常纪律条款（6.4）；复核队列按 `review_reason` + 异常分排序；建立“高异常通话 → 人工确认 → 新规则/新 Few-shot 回写”的迭代 SOP（对应规范第“违规场景涉诈判断与补充”章的人工核查流程）。
  - 验证标准：完成一轮“离线筛查发现 → 规则库新增条目 → 回归通过”的完整闭环演练。

执行顺序建议：1→2→3→4→5→6→7（V1 原序）→ 8→9→10→11。其中 Step 8-9 可与 Step 5-7 并行启动（不依赖在线服务）。

---

## 八、风险与对策（V2 新增项）

| 风险 | 说明 | 对策 |
|---|---|---|
| 冷启动数据不足 | PyOD 需要足量历史语料建模，项目初期可能只有规范中的少量案例 | Step 8-9 延后到积累足够每日质检语料后启动；在此之前 `ANOMALY_ENABLED=false`，系统等同 V1 |
| 异常分被误用为判定依据 | 最大的方案性风险：把统计离群当成违规证据，违反规范的可解释要求 | 第五节融合矩阵的硬约束在**代码后处理**层强制执行（不依赖 LLM 自觉）；测试断言覆盖 |
| 已知违规“不异常” | 高频套路在分布内，异常分低 | 明确异常分不参与下调置信度；已知套路完全由规则通道负责 |
| 高维 embedding 陷阱 | 距离类检测器在高维下失效 | 降维或选 IForest/ECOD；以 ADEngine 的 Pitfall 检查输出为准 |
| 训练-推理偏斜 | 在线特征与离线训练特征口径不一致导致分数失真 | 特征提取与缩放器单一实现、随工件一起版本化序列化 |
| 话术分布漂移 | 诈骗话术随时间演化，旧模型异常分逐渐失准 | 周期重训 + 漂移监控（线上异常分分布对比训练期分布）；`model_version` 字段便于追踪 |
| 在线延迟增加 | embedding + 打分增加单条请求耗时 | 异常打分与规则引擎并行执行；embedding 选轻量本地模型；超时即降级跳过 |
| 误报挤占复核容量 | 异常筛查查准率低会浪费人工 | 阈值按复核容量倒推 top-k% 标定；`anomaly_only` 队列单列统计命中率，持续调阈值 |

---

## 九、修订后的成功标准（Success Criteria V2）

在 V1 成功标准（稳定 JSON 输出；区分正常/违规/涉诈；输出风险等级、类型、说明、证据、复核建议；典型案例可识别）之上，新增：

1. 引入 PyOD 后，黄金样本回归的主判定结果（违规与否、类型、风险等级）与 V1 **零差异**；
2. 离线批量筛查报告可正常产出，top anomalies 人工抽检“值得复核”比例达到约定目标；
3. 至少完成一次由异常筛查发现、人工确认、规则库回写的**新型话术闭环**；
4. `anomaly_only` 复核路径可被规则未覆盖的可疑通话触发；
5. 关闭 `ANOMALY_ENABLED` 或模型工件缺失时，系统行为与 V1 完全一致（降级安全）。

---

## 附：V1 → V2 变更速览

| 维度 | V1（思路.txt） | V2（本方案） |
|---|---|---|
| 判定主链路 | 规则引擎 → LLM 裁决 | 不变 |
| 信号来源 | 规则命中 + LLM | 规则命中 + LLM + **PyOD 异常分（辅助）** |
| 架构形态 | 单一在线管线 | 在线管线 + **离线训练/批量筛查通道** |
| 规则盲区处理 | 回退人工复核（被动） | 异常分主动兜底 + 离线发现新型话术（主动） |
| 新增模块 | — | feature_extractor、anomaly_service、train_anomaly、batch_screen、models 工件 |
| Schema | V1 字段 | 追加 `anomaly` 块与 `review_reason` |
| 依赖 | FastAPI、Pydantic、PyYAML、HTTP 客户端 | 追加 pyod、numpy、scikit-learn、joblib、embedding 依赖 |
| 风险等级判定口径 | 规范规则 | 不变（异常分明确禁止影响） |
| PyOD 集成方式 | — | 直接 Python 调用 ADEngine（预留 MCP 演进） |
