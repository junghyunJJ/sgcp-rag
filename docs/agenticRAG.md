# Agentic RAG 아키텍처

## 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        진입 경로 (2개)                           │
│                                                                 │
│  MCP Tool                          REST API                     │
│  agentic_search()                  POST /collections/{id}/      │
│  (mcp_server.py)                   agentic-search               │
│       │                            (api/agentic.py)             │
│       │         ┌──────────────┐         │                      │
│       └────────►│ run_agentic_ │◄────────┘                      │
│                 │ search()     │                                 │
│                 │ __init__.py  │                                 │
│                 └──────┬───────┘                                 │
│                        │                                        │
│              ┌─────────▼──────────┐                             │
│              │  LangGraph         │                             │
│              │  StateGraph        │                             │
│              │  (graph.py)        │                             │
│              └─────────┬──────────┘                             │
│                        │                                        │
│         ┌──────────────┼──────────────────┐                     │
│         ▼              ▼                  ▼                     │
│    ┌─────────┐   ┌──────────┐     ┌────────────┐              │
│    │ nodes.py│   │graders.py│     │ prompts.py │              │
│    │ (5노드) │   │ (3평가기)│     │ (5프롬프트)│              │
│    └────┬────┘   └──────────┘     └────────────┘              │
│         │                                                      │
│         ▼                                                      │
│    ┌──────────────────────┐                                    │
│    │ Collection.search()  │  ← 기존 코드 재사용 (검색 로직 0% 중복) │
│    │ (database/           │                                    │
│    │  collections.py)     │                                    │
│    └──────────┬───────────┘                                    │
│               │                                                │
│    ┌──────────▼───────────┐                                    │
│    │ get_many_by_source_  │  ← optional wiki source_ref 승격       │
│    │ refs()               │                                    │
│    └──────────┬───────────┘                                    │
│               ▼                                                │
│    ┌──────────────────┐                                        │
│    │  PostgreSQL +    │                                        │
│    │  pgvector        │                                        │
│    └──────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. 데이터 흐름: AgentState

모든 노드가 읽고 쓰는 공유 상태입니다 (`state.py`).

```
AgentState (TypedDict)
├── question          ← 현재 질문 (rewrite 시 갱신됨)
├── collection_id     ← 검색 대상 컬렉션 UUID
├── user_id           ← 접근 제어 (nullable)
├── search_type       ← "semantic" | "keyword" | "hybrid"
├── search_limit      ← 한 번에 가져올 문서 수 (기본 5)
├── search_filter     ← 메타데이터 필터 (nullable)
├── documents         ← retrieve 결과 (raw 검색 결과)
├── relevant_documents← grade 통과한 문서만
├── generation        ← LLM이 생성한 답변
├── query_rewrites    ← 재작성 이력 ["v1", "v2", ...]
├── rewrite_count     ← 현재 루프 카운터
├── max_rewrites      ← 최대 재시도 (기본 3, 루프 방지)
├── use_wiki_context  ← optional LLM Wiki navigation layer 사용 여부
├── wiki_context      ← raw wiki prose는 generation evidence로 사용하지 않음
├── selected_wiki_pages ← 선택된 wiki page metadata (최대 3개)
├── wiki_context_status ← disabled/selected/missing_pack/no_match/invalid_json/invalid_schema
├── wiki_source_refs  ← 선택 page에서 추출한 bounded/deduped source refs
├── wiki_promoted_documents ← source refs로 다시 조회한 실제 chunk들
├── wiki_promotion_status ← disabled/not_selected/promoted/fetch_failed 등
├── steps             ← 실행 추적 로그 ["retrieve: found 5 documents", ...]
└── error             ← 에러 메시지 (nullable)
```

**LangGraph의 State 업데이트 방식**: 각 노드는 전체 state를 반환하지 않고, **변경된 키만 dict로 반환**합니다. 예를 들어 `retrieve`는 `{"documents": [...], "steps": [...]}` 만 반환하면 LangGraph가 기존 state에 merge합니다. 이게 `TypedDict`를 사용하는 이유 -- Pydantic `BaseModel`이면 전체 객체를 매번 재생성해야 합니다.

---

## 2. 그래프 실행 흐름 (graph.py)

```
        ┌─────────┐
        │  START  │
        └────┬────┘
             ▼
     ┌───────────────┐
     │   retrieve    │  Collection.search() 호출
     │   (nodes.py)  │  → documents에 결과 저장
     └───────┬───────┘
             ▼
     ┌───────────────┐
     │grade_documents│  각 문서를 LLM으로 평가
     │   (nodes.py)  │  → relevant_documents에 통과분만
     └───────┬───────┘
             │
      ┌──────┴──────┐  _route_after_grading()
      │             │
 relevant > 0   relevant == 0
      │             │
      ▼             ▼
┌──────────┐  ┌─────────────┐
│ generate │  │rewrite_query│  질문 재작성
│(nodes.py)│  │ (nodes.py)  │  rewrite_count++
└────┬─────┘  └──────┬──────┘
     │               │
     ▼               └──► retrieve로 돌아감 (루프)
┌────────────────┐
│grade_generation│  2단계 검증:
│  (nodes.py)    │  ① 환각 체크
└────┬───────────┘  ② 답변 품질 체크
     │
  ┌──┴──┐  _route_after_generation_check()
  │     │
PASSED  FAILED
  │     │
  ▼     ▼
┌────┐ ┌─────────────┐
│END │ │rewrite_query│ → retrieve로 다시 루프
└────┘ └─────────────┘

⚠️ 루프 방지: rewrite_count >= max_rewrites면
   → 강제 generate 또는 END
```

---

## 3. 각 노드의 역할 (nodes.py)

### 3.1 `retrieve(state)` -- LLM 불필요

```python
# 핵심: 기존 Collection.search()를 그대로 호출
collection = Collection(collection_id=..., user_id=...)
documents = await collection.search(
    question,
    limit=state["search_limit"],      # 기본 5
    search_type=state["search_type"],  # hybrid 추천
    filter=state["search_filter"],
)
```

기존 검색 엔진(semantic/keyword/hybrid)을 100% 재사용합니다.

`use_wiki_context=true`이고 wiki page가 선택되면, graph 시작 전에 `source_refs`가 현재 컬렉션의 실제 chunk로 best-effort 조회됩니다. `retrieve`는 이 promoted chunk를 일반 검색 결과 뒤에 붙인 뒤 `grade_documents`로 넘깁니다. 같은 chunk id가 이미 일반 검색 결과에 있으면 일반 검색 결과가 우선하고 wiki metadata로 덮어쓰지 않습니다.

### 3.2 `grade_documents(state, llm)` -- 문서 필터링

```python
grader = get_document_grader(llm)  # llm.with_structured_output(GradeDocumentRelevance)
for doc in documents:
    result = await grader.ainvoke({"document": doc.content, "question": question})
    if result.binary_score == "yes":
        relevant_docs.append(doc)
```

각 문서를 개별적으로 LLM에게 "이 문서가 질문과 관련 있나요? yes/no"를 물어봅니다.

### 3.3 `generate(state, llm)` -- 답변 생성

```python
context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
# ANSWER_GENERATOR_PROMPT에 question + context를 넣어 답변 생성
result = await chain.ainvoke({"question": question, "context": context})
```

`generate`에는 raw LLM Wiki title/summary/context를 넣지 않습니다. Wiki는 navigation metadata로만 선택되고, 답변에 영향을 줄 수 있는 경로는 `source_refs`가 실제 chunk로 승격되어 `relevant_documents`에 포함된 경우뿐입니다. 환각 검증과 citation 근거도 같은 `relevant_documents`만 사용합니다. 자세한 pack 계약은 [llm-wiki-context.md](llm-wiki-context.md)를 참고합니다.

### 3.4 `rewrite_query(state, llm)` -- 쿼리 재작성

```python
# QUERY_REWRITER_PROMPT로 벡터 검색에 최적화된 질문으로 변환
result = await chain.ainvoke({"question": question})
# "What is LangGraph?" → "LangGraph framework stateful agent architecture"
```

### 3.5 `grade_generation(state, llm)` -- 2단계 검증

```
Stage 1: 환각 체크 (Hallucination Grading)
  "이 답변이 제공된 문서에 근거하고 있나요?" → yes/no
  실패 → rewrite_query로 돌아감

Stage 2: 답변 품질 체크 (Answer Grading)
  "이 답변이 원래 질문을 해결하나요?" → yes/no
  실패 → rewrite_query로 돌아감

  둘 다 통과 → END
```

---

## 4. LLM 바인딩 패턴 (graph.py)

LLM 인스턴스를 graph state에 넣으면 직렬화 문제가 생깁니다. 대신 `functools.partial`로 노드 함수에 LLM을 바인딩합니다. `retrieve`는 LLM이 필요 없으므로 partial 없이 직접 등록합니다. 이 패턴 덕분에 테스트에서 mock LLM을 주입하기도 쉽습니다.

```python
graph.add_node("retrieve", retrieve)                              # LLM 불필요
graph.add_node("grade_documents", partial(grade_documents, llm=llm))
graph.add_node("generate", partial(generate, llm=llm))
graph.add_node("rewrite_query", partial(rewrite_query, llm=llm))
graph.add_node("grade_generation", partial(grade_generation, llm=llm))
```

---

## 5. Graders: Structured Output (graders.py)

```python
class GradeDocumentRelevance(BaseModel):
    binary_score: str  # "yes" | "no"

def get_document_grader(llm):
    structured_llm = llm.with_structured_output(GradeDocumentRelevance)
    prompt = ChatPromptTemplate.from_messages([...])
    return prompt | structured_llm  # LCEL chain
```

`with_structured_output()`은 LLM이 반드시 Pydantic 모델 형태로 응답하도록 강제합니다. JSON 파싱 에러 없이 `result.binary_score`로 바로 접근 가능합니다.

---

## 6. 외부 접근 경로 2개

### REST API (`api/agentic.py`)

```
POST /collections/{collection_id}/agentic-search
Body: { "question": "...", "search_type": "hybrid", "max_rewrites": 3 }
Response: { "generation": "...", "relevant_documents": [...], "steps": [...] }
```

### MCP Tool (`mcp_server.py`, `mcp_sse_server.py`)

```python
@mcp.tool
async def agentic_search(collection_id, question, ...):
    result = await client.request("POST", f"/collections/{id}/agentic-search", ...)
    return json.dumps({"answer": ..., "sources": ..., "steps": ...})
```

MCP는 API를 HTTP로 호출합니다 (직접 Python import 아님).

---

## 7. 기존 시스템과의 관계

```
기존 (유지)                    신규 (추가)
──────────                    ──────────
/documents/search             /agentic-search
  └─ Collection.search()        └─ run_agentic_search()
                                     └─ StateGraph
                                          ├─ retrieve → Collection.search() (재사용!)
                                          ├─ grade_documents (LLM)
                                          ├─ generate (LLM)
                                          ├─ rewrite_query (LLM)
                                          └─ grade_generation (LLM)

search_documents (MCP)        agentic_search (MCP)
multi_query (MCP)               └─ API 호출 → 위 그래프 실행
```

기존 `search_documents`는 raw 문서 청크를 반환하고, `agentic_search`는 **완성된 답변 + 소스 + 실행 추적**을 반환합니다.

---

## 8. 설정 체계 (config.py)

```
우선순위: API 파라미터 > 환경변수 > 기본값

AGENT_LLM_PROVIDER=auto          →  llm_provider 파라미터로 오버라이드 가능
AGENT_LLM_MODEL=qwen3.5:122b     →  auto/ollama에서 우선 사용할 Ollama 모델
AGENT_LLM_OPENAI_MODEL=gpt-5.4   →  auto fallback에서 사용할 OpenAI 모델
AGENT_LLM_TEMPERATURE=0          →  llm_temperature 파라미터로 오버라이드 가능
AGENT_MAX_REWRITES=3             →  max_rewrites 파라미터로 오버라이드 가능

OLLAMA_BASE_URL=http://localhost:5000
AGENT_OLLAMA_BASE_URL=http://localhost:5001
QUERY_EXPANSION_OLLAMA_BASE_URL=http://localhost:5000
QUERY_EXPANSION_LLM_PROVIDER=auto
QUERY_EXPANSION_LLM_MODEL=qwen3.5:35b
QUERY_EXPANSION_OPENAI_MODEL=gpt-5.4
```

요청별로 다른 LLM을 사용할 수 있습니다:

```json
{
  "question": "...",
  "llm_provider": "google",
  "llm_model": "gemini-2.0-flash",
  "llm_temperature": 0.3
}
```

Agentic RAG는 `auto`, `openai`, `google`, `ollama` provider를 지원합니다. `auto`는 `AGENT_OLLAMA_BASE_URL`(없으면 `OLLAMA_BASE_URL`)에서 `AGENT_LLM_MODEL`을 먼저 확인하고 실행 중 Ollama LLM 호출이 실패하면 `AGENT_LLM_OPENAI_MODEL`로 한 번 fallback합니다. 명시적으로 `ollama`를 선택하면 fallback하지 않고 오류를 그대로 반환합니다.

---

## 9. 모듈 구조

```
langconnect/
  agent/                     # Agentic RAG 패키지
    __init__.py              # run_agentic_search() 메인 진입점
    config.py                # LLM 설정 - OpenAI/Google/Ollama 및 Ollama availability
    state.py                 # AgentState TypedDict (26줄)
    prompts.py               # 프롬프트 템플릿 5종 (65줄)
    graders.py               # 문서/환각/답변 평가기 (69줄)
    nodes.py                 # 그래프 노드 함수 5개 (175줄)
    graph.py                 # StateGraph 구성 + 컴파일 (100줄)
  api/
    agentic.py               # POST /agentic-search 엔드포인트 (48줄)
  models/
    agentic.py               # AgenticSearchQuery, AgenticSearchResult (37줄)
```

---

## 10. 테스트 커버리지

| 카테고리 | 테스트 수 | 내용 |
|----------|-----------|------|
| Config | 6 | OpenAI/Google 프로바이더, 환경변수, 파라미터 오버라이드 |
| State | 1 | AgentState 필드 검증 |
| Node 단위 | 8 | retrieve, grade_documents, generate, rewrite_query, grade_generation |
| Graph 라우팅 | 6 | 조건부 엣지 모든 경로 (relevant/no-relevant, pass/fail, max_rewrites) |
| Entry Point | 1 | run_agentic_search 에러 처리 |
| E2E | 2 | Happy path + Rewrite loop |
| **총계** | **24** | **전부 통과** |

테스트 실행: `uv run pytest --confcutdir=tests/unit_tests tests/unit_tests/test_agent_config.py tests/unit_tests/test_agentic_search.py -v`

---

## 향후 확장 (Phase 2+)

> 주 사용 패턴이 MCP(AI 어시스턴트)이므로, 외부 오케스트레이터가 이미 제공하는 기능(웹 검색, 멀티컬렉션 팬아웃, 대화 상태)은 그래프 내부에 중복 구현하지 않는다.

### 구현 예정
- **Streaming 응답** (P2): LangGraph `astream_events()` + FastAPI `StreamingResponse`로 실시간 단계 피드백. 웹 UI 사용 비중 증가 시 구현. `steps` 필드가 이미 incremental 설계됨.
- **Heuristic Query Router** (P3): retrieve 전 ~30줄 휴리스틱으로 검색 타입 자동 선택 (인용구/엔티티→keyword, 자연어→semantic, 기본→hybrid). LLM 호출 없음.

### 보류 (현재 불필요)
- ~~Web Search Fallback~~: MCP 클라이언트가 자체 웹 검색 도구 보유. trust boundary 유지를 위해 그래프 내 미구현.
- ~~Multi-Collection 검색~~: MCP 클라이언트가 순차 호출로 처리 가능. 필요 시 thin wrapper endpoint로 충분.
- ~~PostgreSQL 체크포인터~~: 주 소비자(MCP)가 stateless per-query. 채팅 UI 전환 시 재검토.
