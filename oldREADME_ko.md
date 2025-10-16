# LangConnect Client

<div align="center">

[English README](./README.md) | [한국어 README](./README_ko.md)

</div>

<div align="center">

![Next.js](https://img.shields.io/badge/Next.js-15.3.4-black?style=for-the-badge&logo=next.js)
![React](https://img.shields.io/badge/React-19.0.0-61DAFB?style=for-the-badge&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6+-009688?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql)

**벡터 데이터베이스 관리를 위한 GUI 인터페이스**

![frontend](./assets/frontend-kr.png)

</div>

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [아키텍처](#아키텍처)
- [시작하기](#시작하기)
  - [필수 요구사항](#필수-요구사항)
  - [설치](#설치)
  - [애플리케이션 실행](#애플리케이션-실행)
- [MCP 통합](#mcp-통합)
- [환경 변수](#환경-변수)
- [기여자](#기여자)
- [라이선스](#라이선스)

## 🎯 개요

LangConnect Client는 pgvector 확장이 포함된 PostgreSQL로 구동되는 벡터 데이터베이스 관리를 위한 현대적인 Next.js 기반 GUI 인터페이스입니다. 문서 관리, 벡터 검색 기능, 그리고 Model Context Protocol (MCP)을 통한 AI 어시스턴트와의 원활한 통합을 위한 직관적인 웹 인터페이스를 제공합니다.

이 프로젝트는 [langchain-ai/langconnect](https://github.com/langchain-ai/langconnect)에서 영감을 받았습니다.

## ✨ 주요 기능

### 📚 **컬렉션 관리**
- 사용자 정의 메타데이터 지원을 통한 CRUD 작업
- 실시간 통계 및 대량 작업

### 📄 **문서 관리**
- 다중 형식 지원 (PDF, TXT, MD, DOCX, HTML)
- 자동 텍스트 추출 및 청킹
- 드래그 앤 드롭 배치 업로드

### 🔍 **고급 검색**
- **의미적 검색**: OpenAI 임베딩을 사용한 벡터 유사성 검색
- **키워드 검색**: PostgreSQL 전체 텍스트 검색
- **하이브리드 검색**: 구성 가능한 가중치를 통한 통합 검색

### 🔐 **인증**
- 자동 토큰 갱신을 포함한 Supabase JWT 인증
- 역할 기반 액세스 제어
- NextAuth.js를 통한 안전한 리프레시 토큰 관리

### 🤖 **MCP 통합**
- AI 어시스턴트(Claude, Cursor)를 위한 9개 이상의 도구
- stdio 및 SSE 전송 지원

### 🎨 **현대적인 UI**
- Tailwind CSS가 포함된 Next.js
- 다크/라이트 테마, 다국어 지원 (영어/한국어)

## 🏗️ 아키텍처

![Overall Architecture](./assets/structure.png)

### 인증 플로우

인증 시스템은 안전한 토큰 갱신 메커니즘을 구현합니다:

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│   Browser   │────▶│   NextAuth   │────▶│ Supabase  │
│             │◀────│   (JWT)      │◀────│   Auth    │
└─────────────┘     └──────────────┘     └───────────┘
     │                      │
     │ httpOnly 쿠키       │ 리프레시 토큰
     │ (암호화된 JWT)      │ JWT에 저장됨
     │                      │
     ▼                      ▼
 클라이언트에는          액세스 토큰 만료 시
 accessToken만 노출      자동 갱신
```

**주요 보안 기능:**
- 리프레시 토큰은 클라이언트에 절대 노출되지 않음
- 액세스 토큰 만료 시 자동 토큰 갱신
- 보안 강화를 위한 갱신 시 토큰 로테이션
- httpOnly 쿠키에 암호화된 JWT 저장

## 🚀 시작하기

### 빠른 시작

```bash
# 클론 및 설정
git clone https://github.com/teddynote-lab/langconnect-client.git
cd langconnect-client
cp .env.example .env

# 자격 증명으로 .env 편집 후:
make build   # Docker 이미지 빌드
make up      # 모든 서비스 시작
```

MCP 설정

```bash
make mcp
```

서버 중지

```bash
make down
```

### 필수 요구사항

- Docker 및 Docker Compose
- Node.js 20+ (MCP inspector용)
- UV 패키지 매니저와 Python 3.11+
- Supabase 계정

### 설치

1. **저장소 클론**
   ```bash
   git clone https://github.com/teddynote-lab/langconnect-client.git
   cd langconnect-client
   ```

2. **환경 변수 설정**
   ```bash
   cp .env.example .env
   ```

3. **Supabase 구성**
   
   a. [supabase.com](https://supabase.com)에서 새 프로젝트 생성
   
   b. API 자격 증명 가져오기:
      - Project Settings → API로 이동
      - `URL` 및 `anon public` 키 복사
   
   c. `.env` 파일 업데이트:
   ```env
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-public-key
   ```

4. **애플리케이션 빌드**
   ```bash
   make build
   ```

### 애플리케이션 실행

1. **모든 서비스 시작**
   ```bash
   make up
   ```

2. **서비스 액세스**
   - 🎨 **프론트엔드**: http://localhost:3000
   - 📚 **API 문서**: http://localhost:8080/docs
   - 🔍 **상태 확인**: http://localhost:8080/health

3. **서비스 중지**
   ```bash
   make down
   ```

4. **로그 보기**
   ```bash
   make logs
   ```

## 🤖 MCP 통합

### 자동 설정

1. **MCP 구성 생성**
   ```bash
   make mcp
   ```
   
   이 명령은 다음을 수행합니다:
   - Supabase 자격 증명 입력 요청
   - 자동으로 액세스 토큰 획득
   - 토큰으로 `.env` 업데이트
   - `mcpserver/mcp_config.json` 생성

2. **AI 어시스턴트와 통합**

   **Claude Desktop용:**
   - `mcpserver/mcp_config.json`의 내용 복사
   - Claude Desktop의 MCP 설정에 붙여넣기

   **Cursor용:**
   - MCP 구성 복사
   - MCP 통합 하에 Cursor 설정에 추가

### 사용 가능한 MCP 도구

- `search_documents` - 의미적/키워드/하이브리드 검색 수행
- `list_collections` - 모든 컬렉션 목록 조회
- `get_collection` - 컬렉션 상세 정보 가져오기
- `create_collection` - 새 컬렉션 생성
- `delete_collection` - 컬렉션 삭제
- `list_documents` - 컬렉션의 문서 목록 조회
- `add_documents` - 텍스트 문서 추가
- `delete_document` - 문서 삭제
- `get_health_status` - API 상태 확인
- `multi_query` - 단일 질문에서 여러 검색 쿼리 생성

### RAG 프롬프트 예시

다음은 Claude Desktop에서 사용할 수 있는 RAG 프롬프트 예시입니다.

```markdown
You are a question-answer assistant based on given document.
You must use MCP tool(`langconnect-rag-mcp`) to answer the question.

#Search Configuration:
- Target Collection: (user's request, default value: RAG)
- Search Type: hybrid(preferred)
- Search Limit: 5(default)

#Search Guidelines:
Follow the guidelines step-by-step to find the answer.
1. Use `list_collections` to list up collections and find right **Collection ID** for user's request.
2. Use `multi_query` to generate at least 3 sub-questions which are related to original user's request.
3. Search all queries generated from previous step(`multi_query`) and find useful documents from collection.
4. Use searched documents to answer the question.

---

## Format:
(answer to the question)

**Source**
- [1] (Source and page numbers)
- [2] (Source and page numbers)
- ...

---

[Note]
- Answer in same language as user's request
- Append sources that you've referenced at the very end of your answer.
- If you can't find your answer from <search_results>, just say you can't find any relevant source to answer the question without any narrative sentences.
```

### MCP SSE 서버 실행

#### 스크립트를 사용한 빠른 시작

```bash
# 편리한 런처 스크립트 사용
./run_mcp_sse.sh
```

이 스크립트는 다음을 수행합니다:
- 모든 요구사항 확인 (uv, .env 파일)
- API 서버 실행 여부 확인
- 자동 인증으로 MCP SSE 서버 시작

#### 수동 시작

```bash
# 또는 직접 실행
uv run python mcpserver/mcp_sse_server.py
```

서버에는 이제 자동 인증이 포함됩니다:
- 시작 시 기존 토큰 유효성 테스트
- 토큰이 만료되거나 없는 경우 로그인 요청
- 새 토큰으로 `.env` 자동 업데이트
- 포트 8765에서 SSE 서버 시작

### MCP 통합 테스트

```bash
# MCP Inspector로 테스트
npx @modelcontextprotocol/inspector
```

Inspector에서:
1. 전송 유형으로 "SSE" 선택
2. URL로 `http://localhost:8765/sse` 입력
3. 연결하고 사용 가능한 도구 테스트

## 🔧 환경 변수

| 변수 | 설명 | 필수 |
|----------|-------------|----------|
| `OPENAI_API_KEY` | 임베딩용 OpenAI API 키 | 예 |
| `SUPABASE_URL` | Supabase 프로젝트 URL | 예 |
| `SUPABASE_KEY` | Supabase anon public 키 | 예 |
| `NEXTAUTH_SECRET` | NextAuth.js 시크릿 키 | 예 |
| `NEXTAUTH_URL` | NextAuth URL (기본값: http://localhost:3000) | 예 |
| `NEXT_PUBLIC_API_URL` | 프론트엔드용 공용 API URL | 예 |
| `POSTGRES_HOST` | PostgreSQL 호스트 (기본값: postgres) | 아니오 |
| `POSTGRES_PORT` | PostgreSQL 포트 (기본값: 5432) | 아니오 |
| `POSTGRES_USER` | PostgreSQL 사용자 (기본값: teddynote) | 아니오 |
| `POSTGRES_PASSWORD` | PostgreSQL 비밀번호 | 아니오 |
| `POSTGRES_DB` | PostgreSQL 데이터베이스 이름 | 아니오 |
| `SSE_PORT` | MCP SSE 서버 포트 (기본값: 8765) | 아니오 |

## 👥 기여자

<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/teddylee777">
          <img src="https://avatars.githubusercontent.com/u/10074379?s=400&u=ee37ac1a4bb730df9c80d1ac92311cbbf61c680e&v=4" width="100px;" alt="Teddy Lee"/>
          <br />
          <sub><b>Teddy Lee</b></sub>
        </a>
        <br />
        <a href="https://teddylee777.github.io/" title="Portfolio">🏠</a>
      </td>
      <td align="center">
        <a href="https://github.com/fbwndrud">
          <img src="https://avatars.githubusercontent.com/u/50973794?v=4" width="100px;" alt="fbwndrud"/>
          <br />
          <sub><b>fbwndrud</b></sub>
        </a>
        <br />
        <a href="https://github.com/fbwndrud" title="GitHub">🏠</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/jikime">
          <img src="https://avatars.githubusercontent.com/u/9925165?v=4" width="100px;" alt="jikime"/>
          <br />
          <sub><b>jikime</b></sub>
        </a>
        <br />
        <a href="https://github.com/jikime/next-connect-ui/commits?author=jikime" title="Code">💻</a>
        <a href="https://github.com/jikime/next-connect-ui/pulls?q=is%3Apr+author%3Ajikime" title="Pull Requests">💪</a>
      </td>
      <td valign="top">
        <strong>주요 기여</strong>
        <ul>
          <li>Next-Connect-UI 개발</li>
        </ul>
        <strong>커뮤니티</strong>
        <ul>
          <li><a href="https://www.facebook.com/groups/1183007433518603">Vibe Coding KR 페이스북 그룹</a></li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다 - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

<div align="center">
<a href="https://github.com/teddynote-lab">TeddyNote LAB</a>에서 ❤️로 만들어졌습니다
</div>