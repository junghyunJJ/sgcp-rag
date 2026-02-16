# Agentic RAG 레퍼런스 매핑 분석

## Context

`langconnect/agent/` 의 Agentic RAG 구현이 두 디렉토리의 레퍼런스 노트북들을 어떻게 참고했는지 상세 매핑 분석입니다.

---

## 1. 전체 아키텍처: `07-Adaptive-RAG` 가 핵심 골격

현재 구현의 **자기 수정 루프(self-correcting loop)** 아키텍처는 `07-LangGraph-Adaptive-RAG.ipynb`에서 가장 많이 차용했습니다.

```
07-Adaptive-RAG 흐름:
  route_question → retrieve → grade_documents → decide_to_generate
                                                    ├─ generate → hallucination_check → (loop/END)
                                                    └─ transform_query → retrieve (재시도)

현재 구현 흐름:
  retrieve → grade_documents → _route_after_grading
                                  ├─ generate → grade_generation → (loop/END)
                                  └─ rewrite_query → retrieve (재시도)
```

**차이점**: `route_question` (web vs vectorstore 라우팅)을 제거하고 vectorstore 경로만 유지. 웹 검색은 Phase 2로 남겨둠.

---

## 2. 노드별 상세 매핑

### 2.1 `retrieve` 노드 ← `02-Naive-RAG` + `12-RAG/02-Advanced`

| 레퍼런스 | 구현 (`nodes.py:29-51`) | 차이점 |
|---|---|---|
| `02-Naive-RAG`의 `retrieve_document(state)` — `pdf_retriever.invoke(question)` | `Collection(id, user_id).search()` 호출 | FAISS retriever 대신 기존 pgvector 검색 재사용 |
| `12-RAG/02-Advanced`의 `EnsembleRetriever(BM25+FAISS)` 하이브리드 검색 | `search_type="hybrid"` 파라미터 | 같은 개념이지만 pgvector의 full-text + vector 조합으로 구현 |
| `12-RAG/02-Advanced`의 `MultiQueryRetriever` | 별도 `multi_query` MCP 도구로 이미 존재 | agentic graph와는 분리 유지 |

**핵심 설계 결정**: 새로운 retriever를 만들지 않고 `Collection.search()`를 100% 재사용 → 코드 중복 제로.

### 2.2 `grade_documents` 노드 ← `06-Agentic-RAG` + `03-Groundedness-Check`

| 레퍼런스 | 구현 (`nodes.py:54-78`, `graders.py:18-24`) | 차이점 |
|---|---|---|
| `06-Agentic-RAG`의 `GradeDocuments(BaseModel)` with `binary_score: str` | `GradeDocumentRelevance(BaseModel)` — 동일 `binary_score` 패턴 | 이름만 다름, 구조 동일 |
| `06-Agentic-RAG`의 `llm.with_structured_output(GradeDocuments)` | `llm.with_structured_output(GradeDocumentRelevance)` | 동일 패턴 |
| `03-Groundedness-Check`의 `relevance_check` → `is_relevant` 분기 | `_route_after_grading()` 에서 yes/no 분기 | 동일 이진 분기 로직 |
| `03-Groundedness-Check`의 무한 루프 문제 (`GraphRecursionError`) | `max_rewrites` + `rewrite_count` 루프 가드 | **개선**: 명시적 카운터로 무한 루프 방지 |

**노트북 원본**:
```python
# 06-Agentic-RAG
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")
```

**현재 구현**:
```python
# graders.py
class GradeDocumentRelevance(BaseModel):
    binary_score: str = Field(description="Document relevance: 'yes' or 'no'")
```

### 2.3 `generate` 노드 ← `02-Naive-RAG` + `07-Adaptive-RAG`

| 레퍼런스 | 구현 (`nodes.py:81-104`, `prompts.py:34-44`) | 차이점 |
|---|---|---|
| `02-Naive-RAG`의 `llm_answer` — context + question → LLM | `generate()` — 동일 구조 | 거의 동일 |
| `06-Agentic-RAG`의 `hub.pull("teddynote/rag-prompt")` | `ANSWER_GENERATOR_PROMPT` 자체 프롬프트 | Hub 의존성 제거, 자체 관리 |
| `07-Adaptive-RAG`의 `rag_chain` (prompt \| llm \| StrOutputParser) | `prompt \| llm` 후 `result.content` | StrOutputParser 대신 직접 `.content` 접근 |
| `12-RAG/02-Advanced`의 `format_docs()` | `"\n\n---\n\n".join(...)` | 거의 동일한 문서 포맷팅 |

### 2.4 `rewrite_query` 노드 ← `05-Query-Rewrite` + `06-Agentic-RAG`

| 레퍼런스 | 구현 (`nodes.py:107-134`, `prompts.py:24-32`) | 차이점 |
|---|---|---|
| `05-Query-Rewrite`의 `re_write_prompt` (장문 Steps 포함) | `QUERY_REWRITER_PROMPT` — 간결화 | 의미는 동일, 프롬프트 압축 |
| `05-Query-Rewrite`의 `question: List[str]` + `add_messages` reducer | `query_rewrites: list[str]` 별도 필드 | 노트북은 state에 누적, 구현은 별도 리스트 추적 |
| `06-Agentic-RAG`의 `rewrite()` — semantic intent reasoning | 동일 semantic intent 접근 | 프롬프트 문구만 다름 |
| `05-Query-Rewrite`의 `web_search` fallback | 미구현 | Phase 2 예정 (web search 노드 없음) |

**노트북 원본** (`05-Query-Rewrite`):
```python
re_write_prompt = PromptTemplate(
    template="""Reformulate the given question to enhance its effectiveness
    for vectorstore retrieval. Analyze the original question...(장문)""",
    input_variables=["generation", "question"]
)
```

**현재 구현**:
```python
QUERY_REWRITER_PROMPT = """You are a question re-writer that converts an input question
to a better version optimized for vector store retrieval...(간결)"""
```

### 2.5 `grade_generation` 노드 ← `07-Adaptive-RAG` (2-stage validation)

| 레퍼런스 | 구현 (`nodes.py:137-175`, `graders.py:26-39`) | 차이점 |
|---|---|---|
| `07-Adaptive-RAG`의 `GradeHallucinations(BaseModel)` | `GradeHallucination(BaseModel)` | 이름만 다름 (복수→단수) |
| `07-Adaptive-RAG`의 `GradeAnswer(BaseModel)` | `GradeAnswer(BaseModel)` | **동일** |
| `07-Adaptive-RAG`의 `hallucination_check()` — 2단계(hallucination → answer quality) | `grade_generation()` — 동일 2단계 | 하나의 노드로 통합 |
| `07-Adaptive-RAG`의 3분기 라우팅 (hallucination/relevant/not relevant) | 2분기 (PASSED/FAILED) | 단순화됨 |

**이것이 가장 직접적인 차용**: `07-Adaptive-RAG`의 hallucination grader + answer grader 2단계 검증을 하나의 `grade_generation` 노드에 통합.

---

## 3. 그래프 구조 매핑 (`graph.py`)

### 3.1 StateGraph + TypedDict ← `01-Building-Graphs` + `02-Naive-RAG`

| 레퍼런스 | 구현 (`state.py`, `graph.py`) |
|---|---|
| `01-Building-Graphs`의 `GraphState(TypedDict)` 5개 필드 | `AgentState(TypedDict)` 14개 필드 |
| `02-Naive-RAG`의 `StateGraph(GraphState)` + `compile()` | `StateGraph(AgentState)` + `compile()` — 동일 패턴 |
| `01-Building-Graphs`의 `add_conditional_edges()` | 2개의 conditional edge 사용 — 동일 API |

### 3.2 Conditional Routing ← `03-Groundedness-Check` + `07-Adaptive-RAG`

| 레퍼런스 | 구현 (`graph.py:31-59`) |
|---|---|
| `03-Groundedness-Check`의 `is_relevant()` → "relevant"/"not relevant" | `_route_after_grading()` → "generate"/"rewrite_query" |
| `07-Adaptive-RAG`의 `decide_to_generate()` → "generate"/"transform_query" | 동일 개념, 이름만 다름 |
| `07-Adaptive-RAG`의 `hallucination_check()` → 3분기 | `_route_after_generation_check()` → END/"rewrite_query" |

### 3.3 functools.partial 패턴 (독자적 설계)

```python
graph.add_node("grade_documents", functools.partial(grade_documents, llm=llm))
```

이 패턴은 **레퍼런스 노트북에는 없는 독자적 설계**. 노트북에서는 LLM을 전역 변수로 사용하거나 `model.bind_tools()`를 사용했지만, 프로덕션 코드에서는 `functools.partial`로 LLM을 바인딩하여 LangGraph serialization 문제를 회피.

---

## 4. Structured Output 패턴 ← `06-Agentic-RAG` + `07-Adaptive-RAG`

`graders.py`의 3개 Pydantic 모델은 모두 동일 패턴:

| 노트북 원형 | 구현 |
|---|---|
| `06-Agentic-RAG`의 `GradeDocuments` 1개 모델 | `GradeDocumentRelevance` — 확장 |
| `07-Adaptive-RAG`의 `GradeHallucinations` + `GradeAnswer` | `GradeHallucination` + `GradeAnswer` — 거의 동일 |

**노트북에서 1개 → 구현에서 3개로 확장**: 노트북 06은 문서 관련성만, 노트북 07은 hallucination + answer를 추가. 현재 구현은 세 가지 모두 통합.

---

## 5. RAG 기초 패턴 ← `12-RAG/` 디렉토리

| `12-RAG` 노트북 | 구현에서의 활용 |
|---|---|
| `00-RAG-Basic-PDF` — 8단계 RAG 프레임워크 | 전체 파이프라인의 기본 골격 (Load→Split→Embed→Store→Retrieve→Prompt→LLM→Parse) |
| `02-RAG-Advanced` — EnsembleRetriever (BM25+FAISS) | `Collection.search(type="hybrid")` — pgvector로 구현된 동일 개념 |
| `02-RAG-Advanced` — format_docs() | `nodes.py:90-92`의 문서 join 패턴 |
| `02-RAG-Advanced` — PromptTemplate 패턴 | `prompts.py`의 5개 프롬프트 템플릿 |
| `03-Conversation-With-History` — 대화 이력 관리 | **미적용** (현재 stateless, Phase 2 가능) |

---

## 6. 적용하지 않은 레퍼런스 패턴

| 노트북 패턴 | 미적용 이유 |
|---|---|
| `04-Web-Search`의 TavilySearch fallback | Phase 2 예정 — 현재 vectorstore만 |
| `06-Agentic-RAG`의 `ToolNode` + `tools_condition` | `Collection.search()` 직접 호출로 대체 — 불필요한 Tool 추상화 제거 |
| `06-Agentic-RAG`의 `hub.pull("teddynote/rag-prompt")` | 자체 프롬프트로 관리 — 외부 의존성 제거 |
| `07-Adaptive-RAG`의 `route_question` (web/vectorstore 분기) | 단일 소스(vectorstore)만 지원하므로 불필요 |
| `01-Building-Graphs`의 `MemorySaver` checkpointer | 현재 stateless 설계 — 대화 이력 불필요 |
| `03-Conversation-With-History`의 `RunnableWithMessageHistory` | 동일 이유 |

---

## 7. 요약: 레퍼런스별 기여도

| 레퍼런스 | 기여도 | 주요 차용 |
|---|---|---|
| **`07-Adaptive-RAG`** | ★★★★★ | 전체 자기수정 루프 아키텍처, hallucination/answer grading, 2-stage validation |
| **`06-Agentic-RAG`** | ★★★★☆ | GradeDocuments Pydantic 모델, structured output 패턴, grade_documents 로직 |
| **`05-Query-Rewrite`** | ★★★☆☆ | 쿼리 재작성 프롬프트와 rewrite 노드 구조 |
| **`03-Groundedness-Check`** | ★★★☆☆ | relevance check 이진 분기, 무한 루프 문제 인식 → loop guard 설계 |
| **`02-Naive-RAG`** | ★★☆☆☆ | StateGraph + TypedDict 기본 구조, retrieve → generate 기본 흐름 |
| **`01-Building-Graphs`** | ★★☆☆☆ | LangGraph API 기초 (add_node, add_edge, conditional_edges) |
| **`12-RAG/02-Advanced`** | ★★☆☆☆ | 하이브리드 검색 개념, format_docs 패턴, 프롬프트 구조 |
| **`12-RAG/00-Basic-PDF`** | ★☆☆☆☆ | RAG 8단계 프레임워크 이해 (이미 langconnect에 구현됨) |

### 독자적 설계 요소 (노트북에 없는 것)

1. **`functools.partial` LLM 바인딩** — serialization 안전한 프로덕션 패턴
2. **14필드 AgentState** — 노트북의 3-5필드에서 대폭 확장
3. **명시적 `max_rewrites` 루프 가드** — 노트북의 `recursion_limit` 대신 카운터 기반
4. **`Collection.search()` 재사용** — 노트북의 retriever tool 생성 대신 기존 인프라 활용
5. **3-tier 접근 (REST API + MCP + Graph)** — 노트북에 없는 프로덕션 접근 경로
6. **LLM 설정 외부화** — env var + per-request override 계층 구조

---

## 8. 노드별 코드 비교 (원본 → 구현)

### 8.1 `retrieve` 노드

**노트북 원본** (`02-Naive-RAG` → `retrieve_document`):
```python
def retrieve_document(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    retrieved_docs = pdf_retriever.invoke(latest_question)  # FAISS retriever 직접 호출
    retrieved_docs = format_docs(retrieved_docs)             # 문자열로 포맷팅
    return GraphState(context=retrieved_docs)
```

**현재 구현** (`nodes.py:29-51`):
```python
async def retrieve(state: AgentState) -> dict[str, Any]:
    question = state["question"]
    collection_id = state["collection_id"]
    user_id = state.get("user_id")

    collection = Collection(collection_id=collection_id, user_id=user_id)
    documents = await collection.search(                     # pgvector 검색 재사용
        question,
        limit=state.get("search_limit", 5),
        search_type=state.get("search_type", "hybrid"),      # 하이브리드 검색 지원
        filter=state.get("search_filter"),
    )

    steps = state.get("steps", [])
    steps.append(f"retrieve: found {len(documents)} documents")
    return {"documents": documents, "steps": steps}          # dict 반환 (LangGraph merge)
```

**차이점 분석**:
| 항목 | 노트북 | 구현 |
|---|---|---|
| 반환 타입 | `GraphState(context=...)` 전체 state | `dict` 부분 업데이트 (LangGraph가 merge) |
| retriever | `pdf_retriever.invoke()` (FAISS) | `Collection.search()` (pgvector) |
| 동기/비동기 | 동기 (`def`) | 비동기 (`async def`) |
| 문서 포맷 | `format_docs()`로 즉시 문자열 변환 | raw dict 유지 (나중에 generate에서 포맷) |
| 검색 타입 | similarity 고정 | semantic/keyword/hybrid 선택 가능 |
| 추적 | 없음 | `steps` 리스트에 실행 기록 |

---

### 8.2 `grade_documents` 노드

**노트북 원본 A** (`07-Adaptive-RAG` → `grade_documents`):
```python
def grade_documents(state):
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(                     # 전역 grader 사용
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs}
```

**노트북 원본 B** (`06-Agentic-RAG` → `grade_documents` — 조건부 엣지 함수):
```python
class grade(BaseModel):
    binary_score: str = Field(
        description="Response 'yes' if the document is relevant..."
    )

def grade_documents(state) -> Literal["generate", "rewrite"]:
    model = ChatOllama(temperature=0, model="qwen3:14b")
    llm_with_tool = model.with_structured_output(grade)      # 함수 안에서 LLM 생성

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance...
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n ...""",
    )
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    retrieved_docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": retrieved_docs})
    score = scored_result.binary_score

    if score == "yes":
        return "generate"                                     # 라우팅 결정까지 포함
    else:
        return "rewrite"
```

**현재 구현** (`nodes.py:54-78` + `graders.py:18-49`):
```python
# graders.py — Pydantic 모델 분리
class GradeDocumentRelevance(BaseModel):
    binary_score: str = Field(description="Document relevance: 'yes' or 'no'")

def get_document_grader(llm: BaseChatModel):
    structured_llm = llm.with_structured_output(GradeDocumentRelevance)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing document relevance..."),
        ("human", DOCUMENT_GRADER_PROMPT),
    ])
    return prompt | structured_llm

# nodes.py — 순수 노드 함수
async def grade_documents(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    question = state["question"]
    documents = state.get("documents", [])
    steps = state.get("steps", [])

    grader = get_document_grader(llm)                        # LLM은 partial로 바인딩
    relevant_docs = []

    for doc in documents:
        content = doc.get("page_content", "")
        result = await grader.ainvoke(                       # 비동기 호출
            {"document": content, "question": question}
        )
        if result.binary_score.lower() == "yes":
            relevant_docs.append(doc)

    steps.append(f"grade_documents: {len(relevant_docs)}/{len(documents)} relevant")
    return {"relevant_documents": relevant_docs, "steps": steps}
```

**차이점 분석**:
| 항목 | 07-Adaptive (노트북 A) | 06-Agentic (노트북 B) | 구현 |
|---|---|---|---|
| Pydantic 모델 | `GradeDocuments` (전역) | `grade` (함수 내) | `GradeDocumentRelevance` (별도 파일) |
| LLM 생성 | 전역 변수 | 함수 안에서 매번 생성 | `functools.partial`로 바인딩 |
| 라우팅 | 별도 `decide_to_generate` | 노드가 라우팅까지 담당 | 별도 `_route_after_grading` |
| 프롬프트 | system + human 분리 | PromptTemplate 단일 | system + human 분리 (07 패턴) |
| 문서 접근 | `d.page_content` (Document 객체) | `messages[-1].content` (문자열) | `doc.get("page_content")` (dict) |
| 비동기 | 동기 | 동기 | `async` + `ainvoke` |

**핵심**: 07-Adaptive의 구조(전역 grader + 별도 라우터)를 따르되, 06-Agentic의 `with_structured_output` 패턴을 결합. 프로덕션에서는 관심사 분리(grader 정의 → 노드 로직 → 라우팅)를 더 엄격히 적용.

---

### 8.3 `generate` 노드

**노트북 원본 A** (`07-Adaptive-RAG` → `generate`):
```python
# 전역에서 RAG 체인 생성
prompt = hub.pull("teddynote/rag-prompt")                    # Hub에서 프롬프트 로드
llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0)

def format_docs(docs):
    return "\n\n".join(
        [f'<document><content>{doc.page_content}</content>'
         f'<source>{doc.metadata["source"]}</source>'
         f'<page>{doc.metadata["page"]+1}</page></document>'
         for doc in docs]
    )

rag_chain = prompt | llm | StrOutputParser()

def generate(state):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke(                           # 전역 chain 호출
        {"context": format_docs(documents), "question": question}
    )
    return {"generation": generation}
```

**노트북 원본 B** (`06-Agentic-RAG` → `generate`):
```python
def generate(state):
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content                              # messages에서 추출

    prompt = hub.pull("teddynote/rag-prompt")                # 매번 Hub에서 로드
    llm = ChatOpenAI(model_name=MODEL_NAME, temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}                          # messages에 추가
```

**현재 구현** (`nodes.py:81-104`):
```python
async def generate(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    question = state["question"]
    relevant_docs = state.get("relevant_documents", [])
    steps = state.get("steps", [])

    context = "\n\n---\n\n".join(                            # 간결한 포맷팅
        doc.get("page_content", "") for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_messages([
        ("human", ANSWER_GENERATOR_PROMPT),                  # 자체 프롬프트
    ])
    chain = prompt | llm                                     # StrOutputParser 미사용

    result = await chain.ainvoke({"question": question, "context": context})
    generation = result.content                              # .content 직접 접근

    steps.append("generate: answer produced")
    return {"generation": generation, "steps": steps}
```

**차이점 분석**:
| 항목 | 07-Adaptive | 06-Agentic | 구현 |
|---|---|---|---|
| 프롬프트 소스 | `hub.pull()` (전역) | `hub.pull()` (매번) | `ANSWER_GENERATOR_PROMPT` (자체) |
| 문서 포맷 | XML 태그 + metadata | messages에서 문자열 | `"\n\n---\n\n"` join (간결) |
| 출력 파싱 | `StrOutputParser()` | `StrOutputParser()` | `result.content` 직접 접근 |
| 문서 입력 | `state["documents"]` 전체 | `messages[-1]` | `state["relevant_documents"]` (필터링된 것만) |
| LLM | 전역 변수 | 함수 안 생성 | `partial`로 바인딩 |

**핵심**: Hub 의존성을 제거하고 자체 프롬프트 관리. `relevant_documents`만 사용하여 필터링된 문서와 전체 문서의 명확한 분리.

---

### 8.4 `rewrite_query` 노드

**노트북 원본 A** (`05-Query-Rewrite` → `query_rewrite`):
```python
re_write_prompt = PromptTemplate(
    template="""Reformulate the given question to enhance its effectiveness
    for vectorstore retrieval.

    - Analyze the initial question to identify areas for improvement...
    # Steps
    1. **Understand the Original Question**: ...
    2. **Enhance Clarity**: ...
    3. **Optimize for Retrieval**: ...
    4. **Review**: ...
    # Output Format
    - Provide a single, improved question.
    # Examples
    **Input**: "What are the benefits of..."
    **Output**: "How do renewable energy..."
    # Notes
    [REMEMBER] Re-written question should be in the same language...
    {question}""",
    input_variables=["generation", "question"],              # ← "generation" 미사용
)

question_rewriter = re_write_prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

def query_rewrite(state: GraphState) -> GraphState:
    latest_question = state["question"][-1].content
    question_rewritten = question_rewriter.invoke({"question": latest_question})
    return {"question": question_rewritten}                  # state에 누적 (add_messages)
```

**노트북 원본 B** (`06-Agentic-RAG` → `rewrite`):
```python
def rewrite(state):
    messages = state["messages"]
    question = messages[0].content

    msg = [HumanMessage(
        content=f"""Look at the input and try to reason about
        the underlying semantic intent / meaning.
        Here is the initial question:\n{question}\n
        Formulate an improved question: """,
    )]

    model = ChatOpenAI(temperature=0, model=MODEL_NAME, streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}
```

**노트북 원본 C** (`07-Adaptive-RAG` → `transform_query`):
```python
# 전역 chain (06-Agentic과 거의 동일한 프롬프트)
system = """You a question re-writer that converts an input question to a better
version that is optimized for vectorstore retrieval. Look at the input and try
to reason about the underlying semantic intent / meaning."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])

question_rewriter = re_write_prompt | llm | StrOutputParser()

def transform_query(state):
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question}
```

**현재 구현** (`nodes.py:107-134` + `prompts.py:24-32`):
```python
# prompts.py
QUERY_REWRITER_PROMPT = """\
You are a question re-writer that converts an input question to a better version \
optimized for vector store retrieval. Look at the input and try to reason about \
the underlying semantic intent.

Here is the initial question:
{question}

Formulate an improved question."""

# nodes.py
async def rewrite_query(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    question = state["question"]
    rewrite_count = state.get("rewrite_count", 0)
    query_rewrites = state.get("query_rewrites", [])
    steps = state.get("steps", [])

    prompt = ChatPromptTemplate.from_messages([
        ("human", QUERY_REWRITER_PROMPT),
    ])
    chain = prompt | llm                                     # LLM은 partial로 바인딩

    result = await chain.ainvoke({"question": question})
    new_question = result.content

    query_rewrites.append(new_question)                      # 재작성 이력 추적
    rewrite_count += 1                                       # 카운터 증가
    steps.append(f"rewrite_query: '{question}' -> '{new_question}'")

    return {
        "question": new_question,
        "query_rewrites": query_rewrites,
        "rewrite_count": rewrite_count,
        "steps": steps,
    }
```

**차이점 분석**:
| 항목 | 05-QueryRewrite | 06-Agentic | 07-Adaptive | 구현 |
|---|---|---|---|---|
| 프롬프트 길이 | ~30줄 (Examples 포함) | ~5줄 (인라인) | ~5줄 (system+human) | ~7줄 (07과 유사) |
| 프롬프트 위치 | 전역 변수 | 함수 인라인 | 전역 변수 | 별도 파일 (`prompts.py`) |
| 질문 추적 | `List[str]` + `add_messages` | `messages` 누적 | state 덮어쓰기 | `query_rewrites` 별도 리스트 |
| 루프 제어 | 없음 (무한 루프 가능) | `recursion_limit` 의존 | `recursion_limit` 의존 | **`rewrite_count` + `max_rewrites`** |
| LLM | `gpt-4o-mini` 고정 | `MODEL_NAME` 고정 | 전역 `llm` | 설정 가능 (partial 바인딩) |

**핵심**: 프롬프트는 07-Adaptive와 거의 동일. 05의 장문 프롬프트(Steps/Examples 포함)는 간결화. 가장 큰 차이는 **명시적 루프 가드**(`rewrite_count`/`max_rewrites`) 추가 — 노트북들은 모두 `recursion_limit`에만 의존하여 `GraphRecursionError`로 끝남.

---

### 8.5 `grade_generation` 노드

**노트북 원본** (`07-Adaptive-RAG` → `hallucination_check` — 라우터 겸 검증):
```python
# Pydantic 모델들 (전역)
class GradeHallucinations(BaseModel):
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

class GradeAnswer(BaseModel):
    binary_score: str = Field(
        description="Indicate 'yes' or 'no' whether the answer solves the question"
    )

# grader chains (전역)
hallucination_grader = hallucination_prompt | structured_llm_grader
answer_grader = answer_prompt | structured_llm_grader

def hallucination_check(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Stage 1: Hallucination 체크
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    if grade == "yes":
        # Stage 2: Answer quality 체크
        score = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        grade = score.binary_score

        if grade == "yes":
            return "relevant"                                # 3분기 라우팅
        else:
            return "not relevant"
    else:
        return "hallucination"
```

**현재 구현** (`nodes.py:137-175`):
```python
async def grade_generation(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    generation = state.get("generation", "")
    relevant_docs = state.get("relevant_documents", [])
    question = state["question"]
    steps = state.get("steps", [])

    # Stage 1: Hallucination check (동일 구조)
    documents_text = "\n\n".join(
        doc.get("page_content", "") for doc in relevant_docs
    )
    hallucination_grader = get_hallucination_grader(llm)
    hallucination_result = await hallucination_grader.ainvoke(
        {"documents": documents_text, "generation": generation}
    )

    if hallucination_result.binary_score.lower() != "yes":
        steps.append("grade_generation: FAILED hallucination check")
        return {"steps": steps}                              # 실패 → 바로 반환

    # Stage 2: Answer quality check (동일 구조)
    answer_grader = get_answer_grader(llm)
    answer_result = await answer_grader.ainvoke(
        {"question": question, "generation": generation}
    )

    if answer_result.binary_score.lower() != "yes":
        steps.append("grade_generation: FAILED answer quality check")
        return {"steps": steps}

    steps.append("grade_generation: PASSED both checks")     # 성공 기록
    return {"steps": steps}
```

**차이점 분석**:
| 항목 | 07-Adaptive | 구현 |
|---|---|---|
| 함수 역할 | 검증 + 라우팅 (3분기 반환) | 검증만 (라우팅은 `_route_after_generation_check`) |
| 반환값 | `"hallucination"` / `"relevant"` / `"not relevant"` | `{"steps": [...]}` (steps 문자열로 결과 전달) |
| Pydantic 모델 | `GradeHallucinations` (복수) | `GradeHallucination` (단수) |
| grader 위치 | 전역 변수 | factory 함수 (`get_hallucination_grader(llm)`) |
| 문서 전달 | `documents` (Document 객체 리스트) | `documents_text` (문자열로 join) |
| 2단계 로직 | if-else 중첩 (동일) | if-return 조기 반환 (동일 로직, 다른 스타일) |

**핵심**: 07-Adaptive의 2-stage 검증 로직을 **거의 그대로** 차용. 가장 큰 구조적 차이는 노트북이 "검증 + 라우팅"을 하나의 함수에서 처리하는 반면, 구현은 **검증(grade_generation)과 라우팅(_route_after_generation_check)을 분리**한 점.

---

### 8.6 그래프 빌드 비교

**노트북 원본** (`07-Adaptive-RAG`):
```python
workflow = StateGraph(GraphState)

workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

workflow.add_conditional_edges(
    START, route_question,                                   # web/vectorstore 분기
    {"web_search": "web_search", "vectorstore": "retrieve"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate", hallucination_check,                         # 검증 함수가 3분기 라우터 겸용
    {"hallucination": "generate", "relevant": END, "not relevant": "transform_query"},
)

app = workflow.compile(checkpointer=MemorySaver())
```

**현재 구현** (`graph.py:62-100`):
```python
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve)
graph.add_node("grade_documents", functools.partial(grade_documents, llm=llm))
graph.add_node("generate", functools.partial(generate, llm=llm))
graph.add_node("rewrite_query", functools.partial(rewrite_query, llm=llm))
graph.add_node("grade_generation", functools.partial(grade_generation, llm=llm))

graph.set_entry_point("retrieve")                           # 직접 retrieve 시작
graph.add_edge("retrieve", "grade_documents")

graph.add_conditional_edges(
    "grade_documents", _route_after_grading,                 # 별도 라우터 함수
    {"generate": "generate", "rewrite_query": "rewrite_query"},
)

graph.add_edge("generate", "grade_generation")

graph.add_conditional_edges(
    "grade_generation", _route_after_generation_check,       # 별도 라우터 함수
    {END: END, "rewrite_query": "rewrite_query"},
)

graph.add_edge("rewrite_query", "retrieve")

return graph.compile()                                       # checkpointer 없음
```

**차이점 분석**:
| 항목 | 07-Adaptive | 구현 |
|---|---|---|
| 노드 수 | 5개 (web_search 포함) | 5개 (web_search 제거, grade_generation 추가) |
| 진입점 | `START → route_question` (조건부) | `retrieve` 직접 진입 |
| LLM 바인딩 | 전역 변수 참조 | `functools.partial(node_func, llm=llm)` |
| 라우팅 | 노드 함수가 라우팅 겸용 | 별도 `_route_*` 함수 |
| 분기 수 | generate 후 3분기 | generate 후 2분기 (단순화) |
| checkpointer | `MemorySaver()` | 없음 (stateless) |
| 웹 검색 | `"web_search"` 노드 존재 | 미구현 |

---

### 8.7 Pydantic 모델 + Structured Output 비교

**노트북 원본들**:
```python
# 07-Adaptive-RAG (3개 모델, 모두 전역)
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Indicate 'yes' or 'no' whether the answer solves the question")

# grader 생성 (전역)
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader
```

**현재 구현** (`graders.py`):
```python
# 3개 모델 (별도 파일)
class GradeDocumentRelevance(BaseModel):
    binary_score: str = Field(description="Document relevance: 'yes' or 'no'")

class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Answer grounded in facts: 'yes' or 'no'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses question: 'yes' or 'no'")

# factory 함수 패턴 (LLM을 파라미터로 받음)
def get_document_grader(llm: BaseChatModel):
    structured_llm = llm.with_structured_output(GradeDocumentRelevance)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing document relevance..."),
        ("human", DOCUMENT_GRADER_PROMPT),
    ])
    return prompt | structured_llm
```

**차이점 분석**:
| 항목 | 노트북 | 구현 |
|---|---|---|
| 모델 배치 | 노트북 셀에 인라인 | 별도 `graders.py` 파일 |
| grader 생성 | 전역 변수 (`llm` 고정) | factory 함수 (`llm` 파라미터) |
| Description | 장문 | 간결 |
| 프롬프트 | 인라인 문자열 | 별도 `prompts.py`에서 import |
| 재사용성 | 노트북 내에서만 | 어떤 LLM이든 주입 가능 |

---

### 8.8 State 정의 비교

**노트북 원본들**:
```python
# 06-Agentic-RAG — 최소 state (messages만)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 07-Adaptive-RAG — 3필드
class GraphState(TypedDict):
    question: Annotated[str, "User question"]
    generation: Annotated[str, "LLM generated answer"]
    documents: Annotated[List[str], "List of documents"]

# 05-Query-Rewrite — 5필드
class GraphState(TypedDict):
    question: Annotated[List[str], add_messages]     # ← 리스트로 누적
    context: Annotated[str, "Context"]
    answer: Annotated[str, "Answer"]
    messages: Annotated[list, add_messages]
    relevance: Annotated[str, "Relevance"]
```

**현재 구현** (`state.py` — 14필드):
```python
class AgentState(TypedDict):
    question: str                    # ← 07-Adaptive에서
    collection_id: str               # ← 독자적 (pgvector 컬렉션)
    user_id: str | None              # ← 독자적 (접근 제어)
    search_type: Literal[...]        # ← 독자적 (검색 타입 선택)
    search_limit: int                # ← 독자적
    search_filter: dict | None       # ← 독자적
    documents: list[dict]            # ← 07-Adaptive에서
    relevant_documents: list[dict]   # ← 독자적 (필터링된 문서 분리)
    generation: str                  # ← 07-Adaptive에서
    query_rewrites: list[str]        # ← 05-Query-Rewrite의 List[str] 아이디어
    rewrite_count: int               # ← 독자적 (루프 가드)
    max_rewrites: int                # ← 독자적 (루프 가드)
    steps: list[str]                 # ← 독자적 (실행 추적)
    error: str | None                # ← 독자적 (에러 핸들링)
```

**핵심**: 07-Adaptive의 3필드(`question`, `generation`, `documents`)를 기반으로, 프로덕션에 필요한 11개 필드를 추가. 특히 `relevant_documents`를 `documents`에서 분리한 것이 중요한 설계 결정 — 노트북에서는 `documents`를 필터링 후 덮어쓰지만, 구현에서는 원본(`documents`)과 필터링 결과(`relevant_documents`)를 모두 보존.
