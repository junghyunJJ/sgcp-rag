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

AGENT_LLM_PROVIDER=openai     →  llm_provider 파라미터로 오버라이드 가능
AGENT_LLM_MODEL=gpt-4.1-nano  →  llm_model 파라미터로 오버라이드 가능
AGENT_LLM_TEMPERATURE=0       →  llm_temperature 파라미터로 오버라이드 가능
AGENT_MAX_REWRITES=3           →  max_rewrites 파라미터로 오버라이드 가능
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

---

## 9. 모듈 구조

```
langconnect/
  agent/                     # Agentic RAG 패키지
    __init__.py              # run_agentic_search() 메인 진입점 (97줄)
    config.py                # LLM 설정 - OpenAI/Google 지원 (48줄)
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

- **Adaptive Query Router**: 질문 분석 → 자동으로 semantic/keyword/hybrid 선택
- **Web Search Fallback**: 로컬 문서에 답이 없을 때 Tavily 웹 검색
- **Multi-Collection 검색**: 여러 컬렉션에 걸친 에이전트 검색
- **Streaming 응답**: LangGraph 스트리밍으로 실시간 답변 생성
- **PostgreSQL 체크포인터**: 대화 레벨 상태 영속화
