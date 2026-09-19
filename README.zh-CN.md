# CourseRAG

[English](README.md)

CourseRAG 是一个面向课程资料的 citation-first、closed-corpus RAG 系统。它只使用用户提供的课程材料检索证据；当选定课程没有足够证据时，系统会拒答，而不是用模型的预训练知识补全答案。

## 当前状态

**早期开发 — M5-B 本地产品原型。**

当前项目已经包含文档解析、来源定位、切块、SQLite 元数据、向量索引、课程级检索、检索评估、证据验证、grounded generation 和 citation validation。现在还提供一个无需付费 API 的本地 retrieval-only 工作流：创建课程、上传资料、提问，并查看 Top-5 原始证据。资料列表会显示文件大小并支持删除；默认单文件上限为 50 MiB、单课程总容量为 500 MiB。当前原型还提供基于 httpOnly cookie 的本地注册/登录，并按账号隔离课程；仍不适合直接用于生产环境。

SupportVerifier 已有人工审核的 gold labels，但尚未完成真实语义模型评估。当前原型仍缺少生产级限流、账号恢复和部署加固，暂不应部署给真实用户。

## 核心原则

```text
问题 + 选定课程
  → 只在该课程中检索证据
  → 检查证据是否足够
  → 证据不足或冲突时拒答
  → 证据足够时才允许 grounded generation
  → 校验并返回引用
```

严格 grounding 同时通过后端控制流和模型指令执行。单靠 prompt 不被视为安全边界。

## 架构

- **前端：** Next.js + TypeScript。
- **后端：** FastAPI + Python。
- **应用元数据：** SQLite。
- **向量存储：** Qdrant，并强制使用课程过滤。
- **文档格式：** PDF、PPTX、DOCX、Markdown 和文本。
- **Embedding：** deterministic、FastEmbed 和 OpenAI provider。
- **生成：** OpenAI Responses API provider，只接收通过验证的证据。

完整设计见 [docs/architecture.md](docs/architecture.md) 和 [docs/grounding-policy.md](docs/grounding-policy.md)。

## 已实现功能

- 课程创建和课程级文档元数据。
- PDF 页面、PPTX 幻灯片、DOCX 段落、Markdown 和文本解析。
- 带来源范围的确定性切块。
- SQLite 持久化和同课程重复文档检测。
- Qdrant 向量索引和课程隔离检索。
- Hit@k、Recall@k、MRR、FullEvidence@k 等检索评估。
- `SUPPORTED`、`INSUFFICIENT`、`CONFLICTING` 证据验证结果。
- 仅允许已验证 chunk 的 grounded generation 和 citation validation。
- 本地课程、上传、检索证据和 grounded-generation HTTP API。
- 文件大小展示、课程级上传配额和课程级资料删除。
- 本地注册/登录、会话 cookie 和按账号隔离课程 API。
- 无需付费 API 的 Next.js retrieval-only 界面。

## 快速开始

在仓库根目录执行：

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e "backend[test]"
```

运行后端测试：

```bash
./.venv/bin/pytest -q
```

启动后端：

```bash
./.venv/bin/python -m uvicorn course_rag_api.main:app \
  --app-dir backend/src --host 127.0.0.1 --port 8000 --reload
```

另开终端启动前端：

```bash
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

打开 <http://127.0.0.1:3000>。

如果当前终端没有 `pnpm`，且前端依赖已经存在，可直接运行：

```bash
./node_modules/.bin/next dev
```

本地语义检索可在 `.env` 中设置：

```env
COURSE_RAG_EMBEDDING_PROVIDER=fastembed
```

首次使用会下载 `BAAI/bge-small-en-v1.5` 到被忽略的 `runtime/models/fastembed/`。索引和检索必须使用同一个 embedding provider。retrieval-only 界面不需要 API key；grounded generation 和语义 SupportVerifier 仍需要有额度的 OpenAI API。

## 隐私

真实课程资料、API key、数据库、Qdrant 状态、模型权重和本地评估输出都不应提交到仓库。请勿将真实课程材料上传到公开 repository。运行时数据路径默认位于被 Git 忽略的 `runtime/` 下。

## 计划功能

- SupportVerifier 的真实 live evaluation。
- 更多 embedding 和 generation provider。
- 身份认证、部署加固和对话历史。

这些是计划功能，不代表当前已经实现。
