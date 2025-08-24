export const ko = {
  common: {
    loading: '로딩 중...',
    error: '오류',
    success: '성공',
    cancel: '취소',
    save: '저장',
    delete: '삭제',
    edit: '편집',
    create: '만들기',
    search: '검색',
    refresh: '새로고침',
    logout: '로그아웃',
    none: '없음',
    selectAll: '전체 선택',
    selected: '{{count}}개 선택됨',
    total: '총 {{count}}개',
  },
  sidebar: {
    main: '메인',
    collections: '컬렉션',
    documents: '문서',
    search: '검색',
    apiTester: 'API 테스터',
    mainTitle: '메인'
  },
  collections: {
    title: '컬렉션 관리',
    description: '문서 컬렉션을 생성하고 관리하세요',
    newCollection: '새 컬렉션',
    collectionList: '컬렉션 목록',
    noCollections: '컬렉션이 없습니다',
    noCollectionsDescription: '첫 번째 컬렉션을 생성하여 문서들을 체계적으로 관리해보세요.',
    createFirstCollection: '첫 컬렉션 만들기',
    stats: {
      collections: '컬렉션',
      documents: '문서',
      chunks: '청크',
      documentsCount: '{{count}} 문서',
      chunksCount: '{{count}} 청크'
    },
    table: {
      collection: '컬렉션',
      stats: '통계',
      uuid: 'UUID',
      metadata: '메타데이터'
    },
    deleteConfirm: {
      title: '삭제 확인',
      description: '정말로 선택한 컬렉션을 삭제하시겠습니까? 이 작업은 복구할 수 없습니다.',
      collectionsToDelete: '삭제할 컬렉션 ({{count}}개):',
      warningMessage: '삭제된 컬렉션의 모든 문서도 함께 삭제됩니다.',
      deleteButton: '삭제',
      deleting: '삭제 중...',
      deleteSelected: '선택 항목 삭제'
    },
    popover: {
      basicInfo: '기본 정보',
      statistics: '통계'
    },
    messages: {
      fetchError: '컬렉션 조회 실패',
      deleteSuccess: '{{count}}개의 컬렉션이 성공적으로 삭제되었습니다.',
      deleteFailed: '{{count}}개의 컬렉션 삭제에 실패했습니다.'
    },
    modal: {
      createTitle: '새 컬렉션 만들기',
      nameLabel: '컬렉션 이름',
      namePlaceholder: '컬렉션 이름을 입력하세요',
      descriptionLabel: '설명',
      descriptionPlaceholder: '컬렉션 설명을 입력하세요 (선택사항)',
      creating: '생성 중...',
      createSuccess: '컬렉션이 성공적으로 생성되었습니다',
      createError: '컬렉션 생성 실패'
    }
  },
  documents: {
    title: '문서 관리',
    description: '컬렉션에 문서를 업로드하고 관리하세요',
    selectCollection: '먼저 컬렉션을 선택해주세요',
    uploadDocument: '문서 업로드',
    noDocuments: '문서가 없습니다',
    noDocumentsDescription: '이 컬렉션에 첫 번째 문서를 업로드하세요.',
    uploadFirstDocument: '첫 문서 업로드',
    table: {
      fileName: '파일명',
      uploadDate: '업로드 날짜',
      chunks: '청크',
      actions: '작업'
    },
    deleteConfirm: {
      title: '문서 삭제',
      description: '"{{fileName}}"을(를) 삭제하시겠습니까? 이 작업은 복구할 수 없습니다.',
      warningMessage: '이 문서와 관련된 모든 청크도 함께 삭제됩니다.'
    },
    messages: {
      uploadSuccess: '문서가 성공적으로 업로드되었습니다',
      uploadError: '문서 업로드 실패',
      uploadRequireFiles: '업로드할 파일을 선택하세요',
      deleteSuccess: '문서가 성공적으로 삭제되었습니다',
      deleteError: '문서 삭제 실패',
      fetchError: '문서 조회 실패'
    },
    modal: {
      uploadTitle: '문서 업로드',
      selectFile: '파일 선택',
      supportedFormats: '지원 형식: PDF, TXT, MD, DOCX, HTML',
      uploading: '업로드 중...',
      processing: '문서 처리 중...'
    }
  },
  search: {
    title: '문서 검색',
    description: '시맨틱 또는 키워드 검색을 사용하여 문서를 검색하세요',
    selectCollection: '컬렉션 선택',
    searchPlaceholder: '검색어를 입력하세요...',
    searchButton: '검색',
    searching: '검색 중...',
    searchType: '검색 유형',
    semanticSearch: '시맨틱 검색',
    keywordSearch: '키워드 검색',
    hybridSearch: '하이브리드 검색',
    alphaValue: '알파 값 (0-1)',
    noResults: '검색 결과가 없습니다',
    results: '검색 결과',
    relevanceScore: '관련도: {{score}}',
    enterSearchQuery: '검색어를 입력하세요',
    selectCollectionFirst: '컬렉션을 선택하세요',
    invalidJson: '잘못된 JSON 형식입니다',
    foundResults: '{{count}}개의 결과를 찾았습니다'
  },
  apiTester: {
    title: 'API 테스터',
    description: '인증을 사용하여 API 엔드포인트를 테스트하세요',
    endpoint: '엔드포인트',
    method: '메서드',
    headers: '헤더',
    body: '본문',
    sendRequest: '요청 보내기',
    response: '응답',
    responseTime: '응답 시간: {{time}}ms',
    status: '상태',
    collectionIdRequired: '컬렉션 ID가 필요합니다',
    documentIdRequired: '문서 ID가 필요합니다',
    searchQueryRequired: '검색어를 입력하세요',
    useDocumentUpload: '문서 페이지에서 업로드 기능을 사용하세요',
    goToDocuments: '문서 페이지로 이동 →',
    responsePlaceholder: 'API 응답이 여기에 표시됩니다',
    sendRequestPlaceholder: '요청을 전송하면 응답이 표시됩니다',
    collectionIdPlaceholder: '컬렉션 ID를 입력하세요',
    documentIdPlaceholder: '문서 ID를 입력하세요'
  },
  auth: {
    signIn: '로그인',
    signUp: '회원가입',
    email: '이메일',
    password: '비밀번호',
    confirmPassword: '비밀번호 확인',
    forgotPassword: '비밀번호를 잊으셨나요?',
    alreadyHaveAccount: '이미 계정이 있으신가요?',
    dontHaveAccount: '계정이 없으신가요?',
    signInError: '로그인 실패',
    signInProcessError: '로그인 처리 중 오류가 발생했습니다',
    signInProcessing: '처리 중...',
    signInSuccess: '로그인이 완료되었습니다',
    signInInvalidCredentials: '이메일 또는 비밀번호가 올바르지 않습니다',
    signUpError: '회원가입 실패',
    signUpSuccess: '회원가입이 완료되었습니다',
    signUpEmailExists: '이미 가입된 이메일입니다',
    unknownError: '알 수 없는 오류가 발생했습니다',
    logoutSuccess: '로그아웃되었습니다',
    emailVerification: {
      title: '이메일 확인 필요',
      subtitle: '가입이 거의 완료되었습니다!',
      message: '이메일로 발송된 확인 링크를 클릭하여 이메일 인증을 완료해 주세요.',
      description: '인증이 완료되면 로그인하실 수 있습니다.',
      emailNotReceived: '이메일을 받지 못하셨나요? 스팸 폴더를 확인해 주세요.',
      goToLogin: '로그인 페이지로 이동'
    },
    signUpProcessing: '처리 중...',
    signUpButton: '가입하기',
    signInDescription: '로그인 정보를 입력하세요',
    signUpDescription: '아래 정보를 입력하여 계정을 만드세요'
  },
  theme: {
    light: '라이트',
    dark: '다크',
    system: '시스템'
  },
  language: {
    english: 'English',
    korean: '한국어'
  },
  main: {
    title: '🔗 LangConnect 클라이언트',
    subtitle: '<strong>LangConnect</strong>에 오신 것을 환영합니다.',
    description: 'LangChain과 PostgreSQL로 구동되는 강력한 문서 관리 및 검색 시스템입니다.',
    keyFeatures: '🚀 주요 기능',
    keyFeaturesDescription: '이 애플리케이션은 고급 검색 기능을 갖춘 문서 관리를 위한 포괄적인 인터페이스를 제공합니다:',
    collectionManagement: {
      title: '컬렉션 관리',
      features: [
        '문서 컬렉션 생성 및 관리',
        '컬렉션 통계 보기',
        '컬렉션 일괄 삭제'
      ],
      goTo: '컬렉션으로 이동'
    },
    documentManagement: {
      title: '문서 관리',
      features: [
        '여러 문서 업로드 (PDF, TXT, MD, DOCX)',
        '문서 청크 보기 및 관리',
        '개별 청크 또는 전체 문서 삭제'
      ],
      goTo: '문서로 이동'
    },
    search: {
      title: '검색',
      features: [
        '<strong>시맨틱 검색</strong>: AI 기반 유사도 검색',
        '<strong>키워드 검색</strong>: 전통적인 전문 검색',
        '<strong>하이브리드 검색</strong>: 두 가지 접근법의 장점 결합',
        '고급 메타데이터 필터링'
      ],
      goTo: '검색으로 이동'
    },
    apiTester: {
      title: 'API 테스터',
      features: [
        '모든 API 엔드포인트 직접 테스트',
        'API 기능 탐색',
        '통합 개발 및 디버깅'
      ],
      goTo: 'API 테스터로 이동'
    },
    about: {
      title: '📌 이 프로젝트에 대해',
      description: '<strong>LangConnect</strong>는 다음의 기술들을 결합한 오픈소스 프로젝트입니다:',
      techStack: [
        '<strong>LangChain</strong> - 문서 처리 및 임베딩',
        '<strong>PostgreSQL</strong> - pgvector 확장을 통한 벡터 저장',
        '<strong>FastAPI</strong> - 고성능 API 백엔드',
        '<strong>Streamlit</strong> - 인터랙티브 사용자 인터페이스',
        '<strong>Next.js</strong> - 인터랙티브 사용자 인터페이스'
      ],
      ragReady: 'RAG (Retrieval-Augmented Generation) 애플리케이션 구축에 완벽합니다!',
      links: {
        title: '🔗 링크',
        github: 'GitHub 저장소',
        teddynote: 'TeddyNote LAB',
        docs: '문서',
        nextjsClient: 'Next.js 클라이언트 UI'
      }
    },
    footer: 'Made with ❤️ by'
  }
}