# 벡터 검색 파이프라인 분석

## 1. 검색 시스템 개요

LangConnect는 세 가지 검색 유형을 지원한다: Semantic (벡터 유사도), Keyword (전문 검색), Hybrid (결합 검색). 모든 검색은 `Collection.search()` 메서드에서 처리된다.

```mermaid
flowchart TD
    A["검색 요청<br/>query, search_type, limit, filter"] --> B{"search_type?"}
    B -->|"semantic"| C["벡터 유사도 검색<br/>PGVector.similarity_search_with_score()"]
    B -->|"keyword"| D["전문 검색<br/>PostgreSQL ts_rank + to_tsvector"]
    B -->|"hybrid"| E["하이브리드 검색<br/>Semantic + Keyword 결합"]
    C --> F["백엔드별 메타데이터 필터 적용"]
    D --> F
    E --> F
    F --> G["결과 반환<br/>{id, page_content, metadata, score}"]
```

> **참조**: `langconnect/database/collections.py` 라인 640-907, 검색 API 엔드포인트는 `langconnect/api/documents.py` 라인 226-248

---

## 2. 검색 요청 모델

### 2.1 SearchQuery (요청)

```python
# langconnect/models/document.py (라인 23-27)
class SearchQuery(BaseModel):
    query: str                                                    # 검색 쿼리 문자열
    limit: int | None = 10                                       # 최대 결과 수
    filter: dict[str, Any] | None = None                         # 메타데이터 필터
    search_type: Literal["semantic", "keyword", "hybrid"] = "semantic"  # 검색 유형
```

### 2.2 SearchResult (응답)

```python
# langconnect/models/document.py (라인 30-34)
class SearchResult(BaseModel):
    id: str                                    # 문서 청크 ID
    page_content: str                          # 문서 내용
    metadata: dict[str, Any] | None = None     # 메타데이터
    score: float                               # 관련도 점수 (0~1)
```

### 2.3 API 엔드포인트

```
POST /collections/{collection_id}/documents/search
Content-Type: application/json

{
    "query": "검색할 내용",
    "limit": 10,
    "search_type": "hybrid",
    "filter": {"source": "paper.pdf"}
}
```

> **참조**: `langconnect/api/documents.py` 라인 226-248

---

## 3. Semantic Search (벡터 유사도 검색)

### 3.1 동작 원리

쿼리 텍스트를 임베딩 벡터로 변환한 후, PostgreSQL pgvector의 거리 기반 유사도 검색을 수행한다.

```mermaid
sequenceDiagram
    participant Client as 클라이언트
    participant Col as Collection.search()
    participant Store as PGVector
    participant Embed as HuggingFace 임베딩
    participant PG as PostgreSQL

    Client->>Col: search(query, search_type="semantic")
    Col->>Col: _get_details_or_raise()
    Col->>Store: get_vectorstore(table_id)
    Col->>Store: similarity_search_with_score(query, k=limit)
    Store->>Embed: query → 768차원 벡터
    Embed-->>Store: 쿼리 벡터
    Store->>PG: SELECT ... ORDER BY embedding <=> query_vector LIMIT k
    PG-->>Store: [(Document, distance), ...]
    Store-->>Col: 결과 리스트
    Col->>Col: distance → similarity 변환
    Col->>Col: 메타데이터 필터 적용
    Col-->>Client: [{id, page_content, metadata, score}]
```

### 3.2 거리-유사도 변환

PGVector의 `similarity_search_with_score`는 **거리(distance)**를 반환한다 (낮을수록 유사). 이를 0~1 범위의 유사도 점수로 변환한다:

```
similarity = 1 / (1 + distance)
```

| distance | similarity | 해석 |
|----------|------------|------|
| 0 | 1.0 | 완벽히 동일 |
| 0.5 | 0.667 | 매우 유사 |
| 1.0 | 0.5 | 중간 |
| 2.0 | 0.333 | 낮은 유사도 |
| infinity | 0 | 완전히 다름 |

```python
# langconnect/database/collections.py (라인 699-707)
formatted_results = [
    {
        "id": doc.id,
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "score": 1 / (1 + distance),  # 거리를 유사도로 변환
    }
    for doc, distance in results
]
```

### 3.3 필터와 최소 점수

Semantic 검색은 검증된 metadata filter를 PGVector에 직접 전달한다. 반환된 distance는 similarity로 변환한 뒤 기본 최소 점수(`0.68`)보다 낮은 결과를 제외한다. 호출자는 `min_score`로 이 임계값을 낮추거나 높일 수 있다.

```python
results = store.similarity_search_with_score(
    query,
    k=limit,
    filter=metadata_filter,
)
```

> **참조**: `langconnect/database/collections.py`의 `Collection.search()`

---

## 4. Keyword Search (전문 검색)

### 4.1 동작 원리

PostgreSQL의 전문 검색(Full-Text Search) 기능을 사용한다. `to_tsvector`와 `plainto_tsquery`를 활용하여 텍스트 매칭을 수행하고, `ts_rank`로 관련도를 계산한다.

```mermaid
sequenceDiagram
    participant Client as 클라이언트
    participant Col as Collection.search()
    participant PG as PostgreSQL

    Client->>Col: search(query, search_type="keyword")
    Col->>Col: _get_details_or_raise()
    Col->>PG: SELECT ... WHERE to_tsvector('english', document)<br/>@@ plainto_tsquery('english', query)<br/>ORDER BY ts_rank(...) DESC
    PG-->>Col: [{id, page_content, metadata, score}]
    Col->>Col: 메타데이터 필터 적용
    Col-->>Client: 결과
```

### 4.2 SQL 쿼리 상세

```sql
-- langconnect/database/collections.py (라인 723-740)
SELECT e.id as id,
       e.document as page_content,
       e.cmetadata as metadata,
       ts_rank(
           to_tsvector('english', e.document),
           plainto_tsquery('english', $1)
       ) as score
FROM langchain_pg_embedding e
JOIN langchain_pg_collection c ON e.collection_id = c.uuid
WHERE c.uuid = $2
  AND to_tsvector('english', e.document) @@ plainto_tsquery('english', $1)
ORDER BY score DESC
LIMIT $3
```

**핵심 요소**:

| 함수 | 역할 |
|------|------|
| `to_tsvector('english', document)` | 문서 텍스트를 영어 어간 분석을 적용한 tsvector로 변환 |
| `plainto_tsquery('english', query)` | 검색 쿼리를 평문에서 tsquery로 변환 |
| `@@` 연산자 | tsvector와 tsquery 간 매칭 여부 확인 |
| `ts_rank()` | 매칭된 문서의 관련도 점수 계산 |

**언어 설정**: `'english'`로 고정되어 있어 영어 문서에 최적화되어 있다. 한국어 등 다른 언어의 문서에서는 형태소 분석이 적용되지 않는다.

### 4.3 점수 체계

`ts_rank`는 0~1 범위의 점수를 반환하지만 정확한 범위는 보장되지 않는다. 점수는 쿼리 용어의 문서 내 빈도, 위치 등을 기반으로 계산된다.

> **참조**: `langconnect/database/collections.py` 라인 715-776

---

## 5. Hybrid Search (하이브리드 검색)

### 5.1 전체 흐름

Semantic과 Keyword 검색을 모두 수행한 후, normalized string id를 기준으로 union/dedupe하고 가중 합산으로 최종 점수를 계산한다. 따라서 semantic-only, keyword-only, semantic+keyword 문서가 모두 후보가 될 수 있다.

```mermaid
flowchart TD
    A["검색 쿼리 입력"] --> B["Semantic 검색 실행<br/>fetch_k"]
    A --> C["Keyword 검색 실행<br/>fetch_k"]
    B --> D["결과 결합<br/>(combined_results dict)"]
    C --> D
    D --> E["가중 점수 계산<br/>semantic 70% + keyword 30%"]
    E --> F["점수 기준 내림차순 정렬"]
    F --> G["상위 limit개 반환"]
```

### 5.2 상세 스코어링 알고리즘

#### 단계 1: Semantic 결과 처리

Semantic 검색 결과의 거리를 유사도로 변환한다. 기본 `min_score`보다 낮은 결과는 제외하고, `str(doc.id)`를 결합 key로 사용한다:

```python
for doc, distance in semantic_results:
    similarity_score = _distance_to_similarity(distance)
    if similarity_score < semantic_min_score:
        continue
    doc_id = str(doc.id)
    combined_results[doc_id] = {
        "id": doc_id,
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "semantic_score": similarity_score,
        "keyword_score": 0.0,
    }
```

#### 단계 2: Keyword 결과 처리 및 정규화

Keyword 검색 결과의 `ts_rank` 점수를 최대값 기준으로 0~1 범위로 정규화한다. 같은 id가 semantic 후보에 있으면 `keyword_score`만 채우고, 없으면 keyword-only 후보로 추가한다:

```python
if keyword_rows:
    max_keyword_score = max(
        (float(row["score"]) for row in keyword_rows), default=1.0
    )
    for row in keyword_rows:
        doc_id = str(row["id"])
        normalized_score = (
            float(row["score"]) / max_keyword_score
            if max_keyword_score > 0
            else 0
        )

        if doc_id in combined_results:
            combined_results[doc_id]["keyword_score"] = normalized_score
        else:
            combined_results[doc_id] = {
                "id": doc_id,
                "page_content": row["page_content"],
                "metadata": _metadata_from_row(row),
                "semantic_score": 0.0,
                "keyword_score": normalized_score,
            }
```

#### 단계 3: 최종 점수 계산

최종 score는 모든 후보에 같은 가중합 공식을 적용한다:

```mermaid
flowchart TD
    A["semantic_score"] --> C["score = semantic_score * 0.7 + keyword_score * 0.3"]
    B["keyword_score"] --> C
    C --> D["결과 정렬 & 반환"]
```

```python
for result in combined_results.values():
    score = (
        result["semantic_score"] * HYBRID_SEMANTIC_WEIGHT
        + result["keyword_score"] * HYBRID_KEYWORD_WEIGHT
    )

    all_results.append({
        "id": result["id"],
        "page_content": result["page_content"],
        "metadata": result["metadata"],
        "score": score,
    })
```

### 5.3 스코어링 예시

다음은 가상의 검색 결과에 대한 스코어 계산 예시이다:

| 문서 | semantic_score | keyword_score (정규화) | 유형 | final_score |
|------|---------------|----------------------|------|-------------|
| Doc A | 0.85 | 0.90 | hybrid | **0.865** |
| Doc B | 0.72 | 0 | semantic-only | **0.504** |
| Doc C | 0 | 1.00 | keyword-only | **0.300** |
| Doc D | 0.60 | 0.50 | hybrid | **0.570** |

### 5.4 가중치 설정

| 검색 유형 | 가중치 | 근거 |
|-----------|--------|------|
| Semantic | **70%** | 자연어 쿼리의 의미적 유사도를 우선 |
| Keyword | **30%** | 정확한 용어 매칭을 보완적으로 반영 |

> **참조**: 가중치는 코드에 하드코딩되어 있다 (`langconnect/database/collections.py` 라인 841, 859-862). 현재 동적 조정 기능은 없다.

### 5.5 Over-fetch 전략

Hybrid 검색은 semantic/keyword 양쪽에서 같은 `fetch_k`만큼 후보를 가져온다:

```python
fetch_k = min(max(limit * HYBRID_FETCH_MULTIPLIER, HYBRID_MIN_FETCH_K), HYBRID_MAX_FETCH_K)
semantic_results = store.similarity_search_with_score(query, k=fetch_k)
keyword_rows = await conn.fetch(..., fetch_k)
```

이는 두 검색의 결합과 중복 제거 후에도 충분한 결과를 확보하기 위함이다.

---

## 6. 메타데이터 필터링

### 6.1 필터 적용 방식

Metadata filter는 exact-match scalar 조건만 지원한다. `Collection.search()`는 먼저 필터 shape를 검증하고 `$and` 조건을 평탄화한 뒤, semantic/hybrid의 PGVector 경로와 keyword SQL 경로가 모두 적용할 수 있는 조건만 전달한다.

```python
metadata_filter = _validate_metadata_filter(filter)
store.similarity_search_with_score(query, k=limit, filter=metadata_filter)
```

### 6.2 필터 동작 특성

| 특성 | 설명 |
|------|------|
| 매칭 방식 | 정확 일치 (exact match) |
| 지원 값 | `str`, `int`, `float`, `bool`, `null` |
| 다중 조건 | AND 논리 또는 `$and` 배열 |
| 적용 시점 | PGVector filter 인자와 keyword SQL `jsonb @>` 조건 |
| 지원 key | PGVector-compatible identifier key만 지원 |
| 미지원 key 예시 | `source-url`, `file.name`, 공백/하이픈 포함 key |
| 미지원 shape | `$ne` 같은 연산자, nested object, list |

### 6.3 필터 사용 예시

```json
{
    "query": "machine learning",
    "search_type": "hybrid",
    "filter": {
        "source": "paper.pdf",
        "file_id": "abc123"
    }
}
```

위 필터는 `metadata.source == "paper.pdf"` AND `metadata.file_id == "abc123"`인 결과만 반환한다.

현재 shared semantic/hybrid path는 PGVector 백엔드 제약 때문에 filter field name에 `str.isidentifier()`가 참인 key만 허용한다. 예를 들어 `source-url`, `file.name`, `file name`은 keyword JSONB 조건만으로는 표현 가능하더라도 PGVector-backed search filter와 호환되지 않아 HTTP 400으로 거부된다.

---

## 7. Multi-Query 검색

### 7.1 개요

단일 질문에서 3~5개의 대안 쿼리를 LLM으로 생성하여 검색 커버리지를 확대하는 기능이다. MCP 도구로만 제공된다 (REST API 엔드포인트 없음).

```mermaid
flowchart TD
    A["사용자 질문<br/>'심부전 치료 방법은?'"] --> B["LLM (gpt-5-nano)<br/>3~5개 대안 쿼리 생성"]
    B --> C["대안 쿼리 1<br/>'심부전의 최신 치료법'"]
    B --> D["대안 쿼리 2<br/>'심부전 약물 치료'"]
    B --> E["대안 쿼리 3<br/>'heart failure treatment options'"]
    C --> F["각 쿼리로 search_documents() 호출"]
    D --> F
    E --> F
    F --> G["결과 종합"]
```

### 7.2 구현 상세

```python
# mcpserver/mcp_server.py (라인 730-753)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5-nano", api_key=OPENAI_API_KEY)

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 3 to 5
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Do not number them.
Original question: {question}""",
)

output_parser = LineListOutputParser()
chain = query_prompt | llm | output_parser
queries = await chain.ainvoke({"question": question})
```

### 7.3 LineListOutputParser

LLM 출력을 줄 단위로 분리하여 쿼리 리스트로 변환한다:

```python
# mcpserver/mcp_server.py (라인 92-98)
class LineListOutputParser(BaseOutputParser[list[str]]):
    def parse(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.strip().split("\n")]
        return [line for line in lines if line]
```

### 7.4 RAG 프롬프트

stdio MCP 서버에는 RAG 워크플로우를 안내하는 프롬프트 템플릿이 등록되어 있다:

```python
# mcpserver/mcp_server.py (라인 48-88)
@mcp.prompt("rag-prompt")
async def rag_prompt(query: str) -> list[dict]:
    # 시스템 프롬프트에서 권장하는 검색 가이드라인:
    # 1. list_collections로 컬렉션 찾기
    # 2. multi_query로 3개 이상 하위 질문 생성
    # 3. 생성된 모든 쿼리로 검색 수행
    # 4. 검색 결과로 답변 작성
```

권장 검색 설정:
- **검색 유형**: `hybrid` (선호)
- **검색 제한**: 5개 (기본)

> **참조**: `mcpserver/mcp_server.py` 라인 48-88, 699-757

---

## 8. 검색 파라미터 요약

### 8.1 Collection.search() 매개변수

| 매개변수 | 타입 | 기본값 | 범위 | 설명 |
|----------|------|--------|------|------|
| `query` | `str` | 필수 | - | 검색 쿼리 문자열 |
| `limit` | `int` | `4` | 1~100 | 최대 반환 결과 수 |
| `search_type` | `Literal` | `"semantic"` | semantic, keyword, hybrid | 검색 알고리즘 |
| `filter` | `dict` | `None` | scalar exact match | PGVector-compatible identifier key만 지원 |
| `min_score` | `float \| None` | `None` | 0~1 | semantic 후보 최소 similarity |

> **참조**: `langconnect/database/collections.py` 라인 640-647

### 8.2 MCP search_documents() 매개변수

| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `collection_id` | `str` | 필수 | 컬렉션 UUID |
| `query` | `str` | 필수 | 검색 쿼리 |
| `limit` | `int` | `5` | 최대 결과 수 |
| `search_type` | `str` | `"semantic"` | 검색 유형 |
| `filter_json` | `str \| None` | `None` | 메타데이터 필터 (JSON 문자열) |

> **참조**: `mcpserver/mcp_server.py` 라인 157-165

### 8.3 임베딩 모델 설정

| 항목 | 값 |
|------|-----|
| 모델 | `neuml/pubmedbert-base-embeddings` |
| 벡터 차원 | 768 |
| 디바이스 | CPU |
| 정규화 | L2 정규화 활성화 |
| 거리 측정 | Cosine Distance (PGVector 기본값) |

> **참조**: `langconnect/config.py` 라인 17-26

---

## 9. 검색 유형별 비교

```mermaid
graph LR
    subgraph "Semantic Search"
        S1["쿼리 → 벡터 변환"]
        S2["pgvector 코사인 거리"]
        S3["의미적 유사도 기반"]
    end

    subgraph "Keyword Search"
        K1["쿼리 → tsquery 변환"]
        K2["PostgreSQL FTS"]
        K3["용어 매칭 기반"]
    end

    subgraph "Hybrid Search"
        H1["두 검색 동시 실행"]
        H2["가중 합산 (7:3)"]
        H3["id 기준 union/dedupe"]
    end
```

| 항목 | Semantic | Keyword | Hybrid |
|------|----------|---------|--------|
| **검색 엔진** | PGVector (pgvector) | PostgreSQL FTS | PGVector + PostgreSQL FTS |
| **쿼리 처리** | 벡터 임베딩 변환 | tsquery 변환 | 양쪽 모두 |
| **매칭 방식** | 코사인 거리 유사도 | 어간 분석 텍스트 매칭 | 가중 결합 |
| **언어 지원** | 모든 언어 (임베딩 모델 의존) | 영어 (english 사전 고정) | 양쪽 결합 |
| **장점** | 유의어, 의미 파악 | 정확한 용어 매칭 | 양쪽 장점 결합 |
| **단점** | 정확한 용어 놓칠 수 있음 | 의미적 유사성 미반영 | 2배 연산 비용 |
| **over-fetch** | 없음 | 없음 | `min(max(limit * 4, 20), 100)` |
| **점수 범위** | 0~1 similarity | ts_rank 원본 | weighted sum (semantic 0.7 + keyword 0.3) |
| **권장 사용** | 자연어 질문 | 정확한 키워드 | 종합 검색 (MCP 기본 권장) |

---

## 10. 사용자별 검색 격리

검색 시 `user_id`가 설정된 경우 SQL 쿼리에 소유자 검증 조건이 추가된다:

```sql
-- Keyword/Hybrid 검색 시 사용자 필터 (user_id가 있는 경우)
AND c.cmetadata->>'owner_id' = $3
```

현재 API 엔드포인트에서는 `user_id=None`으로 호출되어 모든 사용자의 데이터를 검색할 수 있다.

> **참조**: `langconnect/database/collections.py` 라인 721-741 (keyword), 라인 785-805 (hybrid)

---

## 11. 검색 결과 형식

### 11.1 REST API 응답

```json
[
    {
        "id": "chunk-uuid-1",
        "page_content": "문서 내용...",
        "metadata": {
            "file_id": "file-uuid",
            "source": "document.pdf",
            "format": "markdown",
            "parser": "PyMuPDF4LLMParser"
        },
        "score": 0.865
    }
]
```

### 11.2 MCP stdio 서버 응답 (JSON 형식)

```json
{
    "results": [
        {
            "content": "문서 내용...",
            "metadata": {"file_id": "...", "source": "..."},
            "score": 0.865,
            "id": "chunk-uuid-1"
        }
    ],
    "count": 1,
    "search_type": "hybrid"
}
```

> **참조**: `mcpserver/mcp_server.py` 라인 201-216. MCP stdio 서버는 `page_content` 대신 `content` 키를 사용한다.

### 11.3 MCP SSE 서버 응답 (Markdown 형식)

```markdown
## Search Results (hybrid)

### Result 1 (Score: 0.8650)
문서 내용...
Document ID: chunk-uuid-1
```

> **참조**: `mcpserver/mcp_sse_server.py` 라인 240-246. SSE 서버는 Markdown 형식의 텍스트를 반환한다.

---

## 12. Hybrid Search Benchmark

Production ranking logic should not be changed only from one-off manual searches.
Use the benchmark script to compare current REST behavior with exploratory
threshold and offline fusion variants.

```bash
uv run python scripts/benchmark_hybrid_search.py \
  --collection-name agentpaper \
  --format table
```

If multiple collections share the same name, pass the UUID directly:

```bash
uv run python scripts/benchmark_hybrid_search.py \
  --collection-id 06bd503e-fd03-4451-8ce7-7c1ee5012584 \
  --format json \
  --output benchmark-results.json
```

Interpretation notes:

- `current_hybrid` is the production REST behavior with the requested `--limit`.
- `current_hybrid_min_070` is an exploratory threshold comparison.
- `offline_fusion_*` lanes are approximate simulations built from public semantic
  and keyword REST responses using `--candidate-limit`; they are not production
  equivalents.
- By default, expectation failures are reported but exit code remains `0`.
  Add `--fail-on-regression` when using the benchmark as a gate.
