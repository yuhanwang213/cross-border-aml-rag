# 资金出境与反洗钱合规问答助手

基于 RAG（检索增强生成）的中文法规问答系统，面向“中国资金出境”和“反洗钱”两类合规问题：回答只依据知识库中的法律法规，每个结论都附“文档名 + 章 + 条”出处；引用由程序逐字校验，找不到依据时明确回答“知识库中没有相关规定”，而不是编造。

除了普通法规问答，系统还支持**场景评估**：用户描述自己打算做的事（如“我想把 80 万人民币换成美元汇给在美国读书的孩子”），系统拆解出其中的合规问题、逐个检索法规与判例，给出“合规可行 / 有条件可行 / 存在较高合规风险 / 违法或被禁止 / 依据不足”的结构化结论。

## 亮点

- **引用可验证**：模型以 JSON 输出答案和逐字引句，程序校验每条引用的来源必须在本次检索结果中、引句必须是条款原文的子串；不合格的引用自动剔除，全部不合格时重试一次，仍无有效引用则拒答（[rag/verify.py](rag/verify.py)）。
- **按法规版式切分**：条款式（第X条，含“第二百八十七条之二”）、提纲式（外汇局通知的“一、二、”）、问答式（答记者问）、案例式（典型案例按【基本案情】【典型意义】拆分），每块都带章节、页码和效力层级（[rag/chunker.py](rag/chunker.py)）。
- **混合检索 + 重排**：bge-m3 稠密向量与稀疏关键词两路召回，RRF 融合后用 bge-reranker-v2-m3 重排，重排分数低于阈值即拒答（[rag/retriever.py](rag/retriever.py)）。
- **领域知识补强**：刑法条文正文里没有罪名（第一百九十一条通篇不出现“洗钱”），系统按最高法罪名规定为条文补上罪名；“走账”“跑分”“地下钱庄”“带现金出国”等口语在检索前扩展为规范术语（[rag/domain.py](rag/domain.py)）。
- **法规与案例分通道**：典型案例只作“实务参考”单独展示，不作为结论依据；场景评估中展示案例的门槛更高。
- **可复现的评测**：调参用评测集与独立检验集分开，每次评测生成逐题的网页报告（[eval/results/](eval/results/)）。

## 架构

```
Regulations/（法律、法规、司法解释、规章、规范性文件、典型案例）
   │  解析：PyMuPDF / python-docx / textutil / 本地 OCR（Apple Vision）
   ▼
按版式切分（条款 / 提纲 / 问答 / 案例）→ data/chunks.jsonl
   │  bge-m3 稠密向量 + 稀疏权重
   ▼
Chroma 向量库 + 稀疏索引
   │
用户提问 ──► 自动识别：法规问答 / 场景评估
   │           场景评估先由大模型拆解成 2～5 个合规子问题
   ▼
口语术语扩展 → 稠密 + 稀疏召回 → RRF 融合 → 重排 → 证据阈值（低于阈值直接拒答）
   ▼
DeepSeek 生成（JSON，必须带 [S编号] 和逐字引句）
   ▼
程序校验引用（来源在检索结果中 + 引句是原文子串）→ 剔除 / 重试 / 拒答
   ▼
Streamlit 界面：答案 + 可展开的条款原文 + 相关典型案例
```

## 知识库

44 份文件、2423 个切块，覆盖外汇管理、个人外汇、境外投资、携带现钞出入境、反洗钱、大额和可疑交易报告、客户尽职调查、刑法相关条文、虚拟货币等主题：

| 效力层级 | 文件数 |
|---|---|
| 法律 | 12 |
| 行政法规 | 4 |
| 司法解释 | 3 |
| 部门规章 | 10 |
| 规范性文件 | 10 |
| 典型案例 | 4 |
| 权威解读（两高答记者问） | 1 |

原文均为公开发布的法律法规和官方案例，放在 [Regulations/](Regulations/)。`Regulations/Internal/` 是早期用来演示权限过滤的虚构企业制度，当前版本不入库。

## 评测结果

| 指标 | 评测集（37 题） | 独立检验集（10 题） |
|---|---|---|
| 检索命中率（Top-8） | 32/32 = 100% | 10/10 = 100% |
| 检索命中率（Top-1） | 20/32 = 62% | 4/10 = 40% |
| 引用准确率 | 32/32 = 100% | 10/10 = 100% |
| 要点覆盖率 | 32/32 = 100% | 10/10 = 100% |
| 拒答正确率（知识库外问题） | 5/5 = 100% | - |
| 误拒率 | 0/32 = 0% | 0/10 = 0% |
| 案例触发准确率 | 27/27 = 100% | - |
| 平均回答长度（问答 / 场景） | 250 / 687 字 | 409 字 |

- **评测集**（[eval/questions.yaml](eval/questions.yaml)）：法规问答 27 题、场景评估 5 题、知识库外 5 题，用于日常调参。
- **独立检验集**（[eval/holdout.yaml](eval/holdout.yaml)）：由需求方按真实用户口吻编写，只用于检验、不参与调参。
- 指标定义见 [scripts/eval.py](scripts/eval.py)。“引用准确率”指回答的引用中包含标准条款；“要点覆盖率”指回答包含参考答案中所有必须出现的关键结论。
- 逐题报告：[评测集](eval/results/report_full_20261005_1238.md)、[独立检验集](eval/results/report_full_20261005_1240.md)（同名 `.html` 为网页版）。数据为 2026-10-05 稳定版的结果。

## 运行

依赖 macOS（`.doc` 解析用系统自带的 `textutil`，扫描件 OCR 用 Apple Vision）。Apple Silicon 上向量化和重排会自动使用 MPS。

```bash
pip install -r requirements.txt
cp .env.example .env          # 填入 DeepSeek API Key

python scripts/build_chunks.py   # 解析 Regulations/ 并切分，输出 data/chunks.jsonl
python scripts/build_index.py    # 向量化并写入 Chroma（首次运行会下载 bge-m3 和重排模型）

streamlit run app.py
```

命令行工具：

```bash
python scripts/search.py "个人每年购汇额度"                    # 只看检索结果，不调用大模型
python scripts/ask.py "我想带3万美元现金出境" --mode scenario -v  # 命令行问答
python scripts/eval.py --retrieval-only                         # 只评检索，附拒答阈值扫描
python scripts/eval.py --questions eval/holdout.yaml            # 完整问答评测（调用 DeepSeek）
```

## 目录

```
app.py              Streamlit 界面
rag/
  parsers.py        文档解析、元数据提取、OCR 与去重
  chunker.py        按版式切分
  domain.py         罪名标注与口语术语扩展
  index.py          写入 Chroma 与稀疏索引
  models.py         bge-m3 / bge-reranker 加载
  retriever.py      混合检索、重排、案例通道
  generator.py      问答与场景评估主流程
  verify.py         引用校验
  config.py         路径与检索、生成参数
scripts/            建库、检索、问答、评测、报告生成
eval/               评测集、独立检验集与历次评测报告
PRD/                需求说明（v1 → v2 的演进）
Regulations/        知识库原文
```

## 技术栈

Python · PyMuPDF · python-docx · ocrmac · FlagEmbedding（BAAI/bge-m3、bge-reranker-v2-m3）· Chroma · DeepSeek（OpenAI 兼容接口）· Streamlit

## 免责声明

本项目是技术演示，回答不构成法律意见。法规可能已修订，实际业务请以现行有效版本和专业意见为准。
