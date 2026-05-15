# Agentic RAG 구현 계획

## Context

LangConnect는 현재 **Advanced RAG** 시스템입니다 (Hybrid Search + Multi-Query). 이를 **Agentic RAG**로 진화시켜, 검색 결과를 자동으로 평가하고, 쿼리를 재작성하며, 답변 품질을 검증하는 에이전트 루프를 추가합니다.

참고 소스: `langchain-kr/17-LangGraph/02-Structures/` 노트북들의 Agentic RAG, Adaptive RAG, Self-RAG 패턴을 기반으로 설계했습니다.

**핵심 원칙**: 기존 검색 기능은 그대로 유지. Agentic 기능은 **추가(additive)** 방식으로 구현.

---

## 아키텍처 개요

```
START → retrieve → grade_documents → [relevant?]
                                        ├─ YES → generate → grade_generation → [passed?]
                                        │                                        ├─ YES → END
                                        │                                        └─ NO → rewrite_query ─┐
                                        └─ NO → rewrite_query ──────────────────────────────────────────┘
                                                      │
                                                      └──→ retrieve (loop, max 3회)
```

기존 `Collection.search()`를 `retrieve` 노드에서 직접 호출 — 검색 로직 중복 없음.

---

## 새 모듈 구조

```
langconnect/
  agent/                     # NEW PACKAGE
    __init__.py              # run_agentic_search() 메인 진입점
    config.py                # LLM 설정 (provider/model 선택)
    state.py                 # AgentState TypedDict
    prompts.py               # 프롬프트 템플릿 5종
    graders.py               # 문서/환각/답변 평가기 (structured output)
    nodes.py                 # 그래프 노드 함수 5개
    graph.py                 # StateGraph 구성 + 컴파일
  api/
    agentic.py               # NEW: /collections/{id}/agentic-search 엔드포인트
  models/
    agentic.py               # NEW: AgenticSearchQuery, AgenticSearchResult
```

---

## Phase 1: Core Agent 모듈 (7 파일 생성)

### 1.1 `langconnect/agent/state.py` — 상태 정의

```python
class AgentState(TypedDict):
    question: str                          # 현재 질문 (재작성 시 갱신)
    collection_id: str                     # 검색 대상 컬렉션
    user_id: str | None                    # 접근 제어
    search_type: Literal["semantic", "keyword", "hybrid"]
    search_limit: int
    search_filter: dict[str, Any] | None
    documents: list[dict[str, Any]]        # 검색 결과
    relevant_documents: list[dict[str, Any]]  # 관련성 통과 문서
    generation: str                        # 생성된 답변
    query_rewrites: list[str]              # 재작성 이력
    rewrite_count: int                     # 루프 카운터
    max_rewrites: int                      # 최대 재시도 (기본 3)
    steps: list[str]                       # 실행 추적
    error: str | None
```

### 1.2 `langconnect/agent/config.py` — LLM 설정

- 환경변수: `AGENT_LLM_PROVIDER` (auto/openai/google/ollama), `AGENT_LLM_MODEL`, `AGENT_LLM_OPENAI_MODEL`, `AGENT_OLLAMA_BASE_URL`, `AGENT_LLM_TEMPERATURE` (0)
- `get_agent_llm(provider, model, temperature)` → `BaseChatModel`
- OpenAI (`ChatOpenAI`), Google (`ChatGoogleGenerativeAI`), Ollama (`ChatOllama`) 지원
- `AGENT_LLM_PROVIDER=auto`일 때는 `AGENT_OLLAMA_BASE_URL`의 Ollama `AGENT_LLM_MODEL`을 먼저 사용하고, Ollama endpoint/model 또는 Ollama LLM 호출이 실패하면 OpenAI `AGENT_LLM_OPENAI_MODEL`로 fallback
- 요청별 오버라이드 가능 (API 파라미터로)

### 1.3 `langconnect/agent/prompts.py` — 프롬프트 5종

| 프롬프트 | 용도 |
|---------|------|
| `DOCUMENT_GRADER_PROMPT` | 검색 문서의 질문 관련성 평가 (yes/no) |
| `QUERY_REWRITER_PROMPT` | 벡터 검색 최적화를 위한 질문 재작성 |
| `ANSWER_GENERATOR_PROMPT` | 관련 문서 기반 답변 생성 |
| `HALLUCINATION_GRADER_PROMPT` | 생성 답변의 근거성 검증 |
| `ANSWER_GRADER_PROMPT` | 답변이 질문을 해결하는지 평가 |

참고 노트북 패턴 그대로 적용 (Pydantic structured output).

### 1.4 `langconnect/agent/graders.py` — 평가기

- `GradeDocumentRelevance(BaseModel)` — binary_score: yes/no
- `GradeHallucination(BaseModel)` — binary_score: yes/no
- `GradeAnswer(BaseModel)` — binary_score: yes/no
- 각각 `llm.with_structured_output()` + 프롬프트 체인

### 1.5 `langconnect/agent/nodes.py` — 5개 노드 함수

| 노드 | 핵심 로직 |
|------|-----------|
| `retrieve(state)` | **기존 `Collection.search()` 직접 호출** — 검색 로직 재구현 없음 |
| `grade_documents(state, llm)` | 각 문서를 LLM으로 관련성 평가, 관련 문서만 필터링 |
| `generate(state, llm)` | 관련 문서로 컨텍스트 구성 → LLM 답변 생성 |
| `rewrite_query(state, llm)` | LLM으로 질문 재작성, rewrite_count 증가 |
| `grade_generation(state, llm)` | 2단계 검증: (1) 환각 체크, (2) 답변 품질 체크 |

### 1.6 `langconnect/agent/graph.py` — StateGraph

- `build_agentic_rag_graph(llm, checkpointer)` → 컴파일된 그래프
- 조건부 엣지 2개:
  - `grade_documents` 후: 관련 문서 있으면 → `generate`, 없으면 → `rewrite_query`
  - `grade_generation` 후: 통과 → `END`, 실패 + 재시도 가능 → `rewrite_query`
- 루프 방지: `rewrite_count >= max_rewrites`이면 강제 `generate` 또는 `END`

### 1.7 `langconnect/agent/__init__.py` — 메인 진입점

- `run_agentic_search(question, collection_id, ...)` → `dict`
- API와 MCP 모두 이 함수를 호출
- 예외 처리 포함 (실패 시 error dict 반환)

---

## Phase 2: API 통합 (3 파일 생성 + 2 파일 수정)

### 2.1 `langconnect/models/agentic.py` — 요청/응답 모델

- `AgenticSearchQuery`: question, search_type, search_limit, filter, max_rewrites, llm_provider, llm_model, llm_temperature
- `AgenticSearchResult`: generation, relevant_documents, steps, query_rewrites, rewrite_count, error

### 2.2 `langconnect/api/agentic.py` — 새 엔드포인트

```
POST /collections/{collection_id}/agentic-search
```
- 기존 `/documents/search`는 그대로 유지
- `run_agentic_search()` 호출

### 2.3 기존 파일 수정 (최소 변경)

**`langconnect/api/__init__.py`** — 1줄 추가:
```python
from langconnect.api.agentic import router as agentic_router
```

**`langconnect/models/__init__.py`** — export 추가

**`langconnect/server.py`** — 1줄 추가:
```python
APP.include_router(agentic_router)
```

---

## Phase 3: MCP 통합 (2 파일 수정)

### 3.1 `mcpserver/mcp_server.py` + `mcp_sse_server.py`

새 MCP 도구 `agentic_search` 추가:
- `search_documents`와 동일한 패턴으로 구현
- API 서버의 `/agentic-search` 엔드포인트 호출
- 반환: answer, sources, steps, rewrites

### 3.2 MCP instructions 및 rag-prompt 리소스 업데이트

기존 4단계 워크플로우에 `agentic_search` 도구 설명 추가

---

## Phase 4: 설정 및 의존성

### 4.1 `pyproject.toml` — 의존성 추가

```toml
"langgraph>=0.2.0",  # 현재 langgraph-sdk만 있음, langgraph 자체 추가
```

### 4.2 `.env.example` — 환경변수 추가

```
AGENT_LLM_PROVIDER=auto
AGENT_LLM_MODEL=qwen3.5:122b
AGENT_LLM_OPENAI_MODEL=gpt-5.4
AGENT_LLM_TEMPERATURE=0
AGENT_MAX_REWRITES=3
OLLAMA_BASE_URL=http://localhost:5000
AGENT_OLLAMA_BASE_URL=http://localhost:5001
QUERY_EXPANSION_OLLAMA_BASE_URL=http://localhost:5000
QUERY_EXPANSION_LLM_PROVIDER=auto
QUERY_EXPANSION_LLM_MODEL=qwen3.5:35b
QUERY_EXPANSION_OPENAI_MODEL=gpt-5.4
```

---

## Phase 5: 테스트

### 5.1 `tests/unit_tests/test_agentic_search.py`

- 상태 초기화 테스트
- 각 노드 단위 테스트 (LLM 목킹)
- 루프 방지 테스트 (max_rewrites=0)
- 전체 그래프 E2E 테스트 (Collection.search + LLM 목킹)
- API 엔드포인트 테스트
- 에러 처리 테스트

### 5.2 `tests/unit_tests/test_agent_config.py`

- OpenAI/Google LLM 생성 테스트
- 환경변수 설정 테스트

---

## 수정 대상 파일 요약

| 파일 | 작업 | 변경 규모 |
|------|------|-----------|
| `langconnect/agent/__init__.py` | 생성 | ~80줄 |
| `langconnect/agent/config.py` | 생성 | ~50줄 |
| `langconnect/agent/state.py` | 생성 | ~30줄 |
| `langconnect/agent/prompts.py` | 생성 | ~60줄 |
| `langconnect/agent/graders.py` | 생성 | ~50줄 |
| `langconnect/agent/nodes.py` | 생성 | ~120줄 |
| `langconnect/agent/graph.py` | 생성 | ~80줄 |
| `langconnect/models/agentic.py` | 생성 | ~40줄 |
| `langconnect/api/agentic.py` | 생성 | ~50줄 |
| `langconnect/api/__init__.py` | 수정 | +2줄 |
| `langconnect/models/__init__.py` | 수정 | +5줄 |
| `langconnect/server.py` | 수정 | +2줄 |
| `mcpserver/mcp_server.py` | 수정 | +60줄 |
| `mcpserver/mcp_sse_server.py` | 수정 | +60줄 |
| `pyproject.toml` | 수정 | +1줄 |
| `.env.example` | 수정 | +4줄 |
| `tests/unit_tests/test_agentic_search.py` | 생성 | ~150줄 |
| `tests/unit_tests/test_agent_config.py` | 생성 | ~50줄 |

**총 18개 파일 (11 생성 + 7 수정), 약 900줄**

---

## 검증 계획

1. **단위 테스트**: `uv run pytest tests/unit_tests/test_agentic_search.py -v`
2. **API 테스트**: `curl -X POST http://localhost:8888/collections/{id}/agentic-search -d '{"question": "테스트 질문"}'`
3. **MCP 테스트**: `npx @modelcontextprotocol/inspector`로 `agentic_search` 도구 호출
4. **E2E 테스트**: Docker 환경에서 `make up` 후 프론트엔드 → API → 에이전트 → DB 전체 흐름 확인

---

## 향후 확장 (Phase 2+)

- **Adaptive Query Router**: 질문 분석 → 자동으로 semantic/keyword/hybrid 선택
- **Web Search Fallback**: 로컬 문서에 답이 없을 때 Tavily 웹 검색
- **Multi-Collection 검색**: 여러 컬렉션에 걸친 에이전트 검색
- **Streaming 응답**: LangGraph 스트리밍으로 실시간 답변 생성
- **PostgreSQL 체크포인터**: 대화 레벨 상태 영속화
