# Hybrid Search Score 의미 불일치 수정

> **수정일**: 2025-01-31
> **영향 범위**: `langconnect/database/collections.py`
> **Breaking Change**: 기존 검색 결과의 score 값이 변경됨

## 문제 진단

### 핵심 버그: Score 의미가 반대

LangChain의 `similarity_search_with_score` 메서드는 **Distance(거리)**를 반환합니다. 이는 의도된 동작입니다.

| 검색 타입 | 반환값 의미 | 좋은 값 | 기존 코드 가정 |
|----------|------------|--------|---------------|
| **Semantic** (`similarity_search_with_score`) | **Distance** | **낮을수록** 유사 | ❌ 높을수록 좋다고 가정 |
| **Keyword** (`ts_rank`) | **Relevance** | **높을수록** 유사 | ✅ 올바름 |

### 근거

- [LangChain GitHub Issue #13437](https://github.com/langchain-ai/langchain/issues/13437): "scores returned are NOT proportional to similarity"
- [LangChain JS Issue #9782](https://github.com/langchain-ai/langchainjs/issues/9782): "does not return a score, but the distance"

### 기존 코드의 문제

```python
# 문제: distance를 높을수록 좋다고 가정하고 정규화
max_semantic_score = max((score for _, score in semantic_results), default=1.0)
normalized_score = score / max_semantic_score  # 거리가 클수록 1에 가까워짐 (잘못됨!)
combined_score = normalized_score * 0.7
```

**결과**: 가장 유사한 문서(distance=0에 가까운)가 가장 낮은 점수를 받음

---

## 수정 내용

### 변환 공식

```python
similarity = 1 / (1 + distance)
```

| Distance | Similarity | 설명 |
|----------|------------|------|
| 0.00 | 1.0000 | Perfect match |
| 0.10 | 0.9091 | Very similar |
| 0.50 | 0.6667 | Moderately similar |
| 1.00 | 0.5000 | Somewhat dissimilar |
| 2.00 | 0.3333 | Dissimilar |

### 수정 1: Semantic Search (lines 696-706)

**Before:**
```python
formatted_results = [
    {
        "id": doc.id,
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "score": score,
    }
    for doc, score in results
]
```

**After:**
```python
# Note: similarity_search_with_score returns DISTANCE, not similarity
# Lower distance = more similar. Convert to similarity: 1 / (1 + distance)
# This gives: distance=0 → similarity=1, distance=∞ → similarity=0
formatted_results = [
    {
        "id": doc.id,
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "score": 1 / (1 + distance),  # Convert distance to similarity
    }
    for doc, distance in results
]
```

### 수정 2: Hybrid Search (lines 829-841)

**Before:**
```python
max_semantic_score = max(
    (score for _, score in semantic_results), default=1.0
)
for doc, score in semantic_results:
    normalized_score = (
        score / max_semantic_score if max_semantic_score > 0 else 0
    )
    combined_results[doc.id] = {
        ...
        "semantic_score": normalized_score,
        "combined_score": normalized_score * 0.7,
    }
```

**After:**
```python
# Add semantic results with distance-to-similarity conversion
# Note: similarity_search_with_score returns DISTANCE, not similarity
# Lower distance = more similar. Convert to similarity: 1 / (1 + distance)
for doc, distance in semantic_results:
    similarity_score = 1 / (1 + distance)
    combined_results[doc.id] = {
        ...
        "semantic_score": similarity_score,
        "combined_score": similarity_score * 0.7,
    }
```

---

## 대안 검토

| 옵션 | 변환 공식 | 장점 | 단점 |
|------|----------|------|------|
| **A. `1 / (1 + distance)`** ✅ | 직접 계산 | 명확, 제어 가능, 항상 (0,1] 범위 | LangChain 내부 로직과 다름 |
| **B. `1.0 - distance`** | LangChain 공식 | 공식 권장, 일관성 | distance>1일 때 음수 가능 |
| **C. `similarity_search_with_relevance_scores`** | 내장 메서드 | 자동 변환 | 성능 약간 저하, hybrid에 통합 어려움 |

**결정**: 옵션 A (`1 / (1 + distance)`) 사용
- 항상 (0, 1] 범위 보장 (음수 불가능)
- 수학적으로 명확하고 예측 가능
- Hybrid search에서 keyword score와 자연스럽게 융합

---

## Hybrid Score Fusion 예시

```
Input:
  Semantic: [(doc1, distance=0.1), (doc2, distance=0.5), (doc3, distance=1.5)]
  Keyword:  [(doc1, ts_rank=0.8), (doc2, ts_rank=0.3), (doc4, ts_rank=0.9)]

Output (sorted by combined score):
  Doc ID |   Semantic |    Keyword |   Combined
  -------|------------|------------|------------
    doc1 |     0.9091 |     0.8889 |     0.9030
    doc2 |     0.6667 |     0.3333 |     0.5667
    doc4 |     0.0000 |     1.0000 |     0.3000
    doc3 |     0.4000 |     0.0000 |     0.2800

✓ doc1 has highest combined score (strong semantic + keyword)
✓ All scores in [0, 1] range
✓ Higher score = more relevant document
```

---

## 영향 범위

### 변경된 API 응답

- `Collection.search()` 메서드의 semantic 및 hybrid 검색
- MCP 서버의 `search_documents` 도구
- API endpoint `/collections/{id}/documents/search`

### Breaking Change

기존 검색 결과의 score 값이 변경됩니다:
- **이전**: Distance 값 그대로 반환 (낮을수록 유사)
- **이후**: Similarity 값 반환 (높을수록 유사)

클라이언트에서 score 값을 기반으로 필터링하거나 정렬하는 경우, 로직 수정이 필요할 수 있습니다.

---

## 검증 방법

### MCP 도구로 실제 검색 테스트

```python
# 동일한 쿼리로 세 가지 검색 타입 비교
search_documents(collection_id, query, search_type="semantic")
search_documents(collection_id, query, search_type="keyword")
search_documents(collection_id, query, search_type="hybrid")

# 확인 사항:
# - 모든 score가 0~1 범위
# - 더 관련 있는 문서가 더 높은 score
# - hybrid score가 semantic과 keyword의 가중 합산 (70:30)
```

### 수동 검증

1. 특정 문서의 정확한 키워드로 검색
2. semantic/keyword/hybrid 모두에서 해당 문서가 상위 랭크
3. score 값이 논리적으로 일관성 있는지 확인

---

## LangChain 최신 동향 (2025년 1월 기준)

### 공식 동작 확인

| 메서드 | 반환값 | 범위 | 해석 |
|--------|--------|------|------|
| `similarity_search_with_score` | **Distance** | [0, 2] (cosine) | 낮을수록 유사 |
| `similarity_search_with_relevance_scores` | **Similarity** | [0, 1] | 높을수록 유사 |

### 관련 GitHub Issues

- **[Issue #13437](https://github.com/langchain-ai/langchain/issues/13437)**: Closed as not planned - 의도된 동작
- **[Issue #17333](https://github.com/langchain-ai/langchain/issues/17333)**: Cosine distance가 1을 초과할 수 있음
- **[langchain-postgres #234](https://github.com/langchain-ai/langchain-postgres/issues/234)**: Hybrid search score fusion 문제

### langchain-postgres 업데이트

- **v0.0.14+**: `PGVector` → `PGVectorStore` 마이그레이션 권장
- **v0.0.16**: 최신 릴리스 (2025년 10월)
- `similarity_search_with_score`의 distance 반환 동작은 변경되지 않음
