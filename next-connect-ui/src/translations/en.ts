export const en = {
  common: {
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    create: 'Create',
    search: 'Search',
    refresh: 'Refresh',
    none: 'None',
    selectAll: 'Select all',
    selected: '{{count}} selected',
    total: 'Total {{count}}',
  },
  sidebar: {
    main: 'Main',
    collections: 'Collections',
    documents: 'Documents',
    search: 'Search',
    apiTester: 'API Tester',
    mainTitle: 'Main'
  },
  collections: {
    title: 'Collection Management',
    description: 'Create and manage document collections',
    newCollection: 'New Collection',
    collectionList: 'Collection List',
    noCollections: 'No collections',
    noCollectionsDescription: 'Create your first collection to organize documents systematically.',
    createFirstCollection: 'Create First Collection',
    stats: {
      collections: 'Collections',
      documents: 'Documents',
      chunks: 'Chunks',
      documentsCount: '{{count}} documents',
      chunksCount: '{{count}} chunks'
    },
    table: {
      collection: 'Collection',
      stats: 'Statistics',
      uuid: 'UUID',
      metadata: 'Metadata',
      wiki: 'Wiki'
    },
    deleteConfirm: {
      title: 'Delete Confirmation',
      description: 'Are you sure you want to delete the selected collections? This action cannot be undone.',
      collectionsToDelete: 'Collections to delete ({{count}}):',
      warningMessage: 'All documents in the deleted collections will also be deleted.',
      deleteButton: 'Delete',
      deleting: 'Deleting...',
      deleteSelected: 'Delete Selected'
    },
    popover: {
      basicInfo: 'Basic Information',
      statistics: 'Statistics'
    },
    messages: {
      fetchError: 'Failed to fetch collections',
      deleteSuccess: '{{count}} collections successfully deleted.',
      deleteFailed: '{{count}} collections failed to delete.'
    },
    modal: {
      createTitle: 'Create New Collection',
      nameLabel: 'Collection Name',
      namePlaceholder: 'Enter collection name',
      descriptionLabel: 'Description',
      descriptionPlaceholder: 'Enter collection description (optional)',
      creating: 'Creating...',
      createSuccess: 'Collection created successfully',
      createError: 'Failed to create collection'
    }
  },
  wiki: {
    title: 'LLM Wiki',
    backToCollections: 'Back to Collections',
    generatedNavigationNotice: 'Generated wiki navigation. Use original retrieved chunks as evidence.',
    index: 'Index',
    sources: 'Sources',
    concepts: 'Concepts',
    noSources: 'No sources',
    noConcepts: 'No concepts',
    notGeneratedTitle: 'Wiki not generated yet',
    notGeneratedDescription: 'Build generated navigation for this collection from the current documents.',
    rebuild: 'Rebuild Wiki',
    rebuilding: 'Rebuilding...',
    loadError: 'Failed to load LLM Wiki',
    pageLoadError: 'Failed to load wiki page',
    rebuildSuccess: 'Wiki rebuilt successfully',
    rebuildError: 'Failed to rebuild wiki',
  },
  documents: {
    title: 'Document Management',
    description: 'Upload and manage documents in collections',
    selectCollection: 'Please select a collection first',
    uploadDocument: 'Upload Document',
    noDocuments: 'No documents',
    noDocumentsDescription: 'Upload your first document to this collection.',
    uploadFirstDocument: 'Upload First Document',
    table: {
      fileName: 'File Name',
      uploadDate: 'Upload Date',
      chunks: 'Chunks',
      actions: 'Actions'
    },
    deleteConfirm: {
      title: 'Delete Document',
      description: 'Are you sure you want to delete "{{fileName}}"? This action cannot be undone.',
      warningMessage: 'All chunks associated with this document will also be deleted.'
    },
    messages: {
      uploadSuccess: 'Document uploaded successfully',
      uploadError: 'Failed to upload document',
      uploadRequireFiles: 'Please select files to upload',
      deleteSuccess: 'Document deleted successfully',
      deleteError: 'Failed to delete document',
      fetchError: 'Failed to fetch documents'
    },
    modal: {
      uploadTitle: 'Upload Document',
      selectFile: 'Select File',
      supportedFormats: 'Supported formats: PDF, TXT, MD, DOCX, HTML',
      uploading: 'Uploading...',
      processing: 'Processing document...'
    }
  },
  search: {
    title: 'Document Search',
    description: 'Search documents using semantic or keyword search',
    selectCollection: 'Select Collection',
    searchPlaceholder: 'Enter search query...',
    searchButton: 'Search',
    searching: 'Searching...',
    searchType: 'Search Type',
    semanticSearch: 'Semantic Search',
    keywordSearch: 'Keyword Search',
    hybridSearch: 'Hybrid Search',
    alphaValue: 'Alpha value (0-1)',
    noResults: 'No results found',
    results: 'Search Results',
    relevanceScore: 'Relevance: {{score}}',
    enterSearchQuery: 'Please enter search query',
    selectCollectionFirst: 'Please select a collection',
    invalidJson: 'Invalid JSON format',
    foundResults: 'Found {{count}} results',
    agenticSearch: 'Agentic Search',
    standardSearch: 'Standard Search',
    agenticDescription: 'AI-powered search with automatic query refinement and answer generation',
    askQuestion: 'Ask a question...',
    maxRewrites: 'Max Query Rewrites',
    aiAnswer: 'AI Answer',
    sourceDocuments: 'Source Documents',
    executionSteps: 'Execution Steps',
    queryRewrites: 'Query Rewrites',
    agenticSearching: 'AI is analyzing your question...',
    agenticNoAnswer: 'Could not generate an answer. Try rephrasing your question.',
    cancelSearch: 'Cancel',
    rewriteHistory: 'Original → Rewritten',
    stepRetrieve: 'Retrieving documents',
    stepGrade: 'Grading relevance',
    stepGenerate: 'Generating answer',
    stepRewrite: 'Rewriting query',
    stepCheck: 'Validating answer',
  },
  apiTester: {
    title: 'API Tester',
    description: 'Test API endpoints with authentication',
    endpoint: 'Endpoint',
    method: 'Method',
    headers: 'Headers',
    body: 'Body',
    sendRequest: 'Send Request',
    response: 'Response',
    responseTime: 'Response time: {{time}}ms',
    status: 'Status'
    ,
    collectionIdRequired: 'Collection ID is required',
    documentIdRequired: 'Document ID is required',
    searchQueryRequired: 'Search query is required',
    useDocumentUpload: 'Please use document upload in Documents page',
    goToDocuments: 'Go to Documents →',
    collectionIdPlaceholder: 'Enter collection ID',
    documentIdPlaceholder: 'Enter document ID',
    responsePlaceholder: 'API response will appear here',
    sendRequestPlaceholder: 'Send a request to see the response'
  },
  theme: {
    light: 'Light',
    dark: 'Dark',
    system: 'System'
  },
  language: {
    english: 'English',
    korean: '한국어'
  },
  main: {
    title: '🔗 LLM Wiki\n-RAG',
    subtitle: 'Welcome to <strong>LLM Wiki-RAG</strong>.',
    description: 'A powerful document management and search system powered by LangChain and PostgreSQL.',
    keyFeatures: '🚀 Key Features',
    keyFeaturesDescription: 'This application provides a comprehensive interface for document management with advanced search capabilities:',
    collectionManagement: {
      title: 'Collection Management',
      features: [
        'Create and manage document collections',
        'View collection statistics',
        'Batch delete collections'
      ],
      goTo: 'Go to Collections'
    },
    documentManagement: {
      title: 'Document Management',
      features: [
        'Upload multiple documents (PDF, TXT, MD, DOCX)',
        'View and manage document chunks',
        'Delete individual chunks or entire documents'
      ],
      goTo: 'Go to Documents'
    },
    search: {
      title: 'Search',
      features: [
        '<strong>Semantic Search</strong>: AI-powered similarity search',
        '<strong>Keyword Search</strong>: Traditional full-text search',
        '<strong>Hybrid Search</strong>: Combines benefits of both approaches',
        'Advanced metadata filtering'
      ],
      goTo: 'Go to Search'
    },
    apiTester: {
      title: 'API Tester',
      features: [
        'Test all API endpoints directly',
        'Explore API functionality',
        'Integration development and debugging'
      ],
      goTo: 'Go to API Tester'
    },
    about: {
      title: '📌 About This Project',
      description: '<strong>LLM Wiki-RAG</strong> is an open-source project that combines the following technologies:',
      techStack: [
        '<strong>LangChain</strong> - Document processing and embeddings',
        '<strong>PostgreSQL</strong> - Vector storage with pgvector extension',
        '<strong>FastAPI</strong> - High-performance API backend',
        '<strong>Streamlit</strong> - Interactive user interface',
        '<strong>Next.js</strong> - Interactive user interface'
      ],
      ragReady: 'Perfect for building RAG (Retrieval-Augmented Generation) applications!',
      links: {
        title: '🔗 Links',
        github: 'GitHub Repository',
        teddynote: 'TeddyNote LAB',
        docs: 'Documentation',
        nextjsClient: 'Next.js Client UI'
      }
    },
    footer: 'Made with ❤️ by'
  }
}
