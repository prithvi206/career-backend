# LangSmith Tracing Setup for Career RAG System

This document explains how to set up LangSmith tracing to monitor and debug your RAG system.

## Prerequisites

1. Sign up for a LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Get your LangSmith API key from the dashboard

## Environment Variables

Add these environment variables to your `.env` file:

```bash
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# LangSmith Configuration for tracing
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=career-rag-system
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

## What Gets Traced

The following operations are automatically traced:

### 1. Document Processing

- **process_document**: Traces PDF/TXT document loading and processing
- **chunk_documents**: Traces document chunking with text splitter
- **add_documents_to_vector_store**: Traces document embedding and storage

### 2. Query Processing

- **query_career_info**: Main RAG query processing with detailed tracing
- **career_rag_query**: Detailed trace showing:
  - Original query and enhanced query
  - Retrieved documents and their metadata
  - Final AI response
  - Document retrieval count and context

### 3. API Endpoints

- **upload_document_endpoint**: Traces document upload flow
- **query_career_endpoint**: Traces career query API calls

## Viewing Traces

1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project (default: "career-rag-system")
3. View traces in real-time as you use the system

## Trace Information Available

- **Input/Output**: See exactly what goes into and comes out of each step
- **Retrieval Results**: View which documents were retrieved for each query
- **Token Usage**: Monitor LLM token consumption
- **Performance**: See timing for each operation
- **Error Traces**: Debug failures with full context

## Disabling Tracing

To disable tracing, set:

```bash
LANGCHAIN_TRACING_V2=false
```

Or remove the `LANGCHAIN_API_KEY` from your environment.

## Troubleshooting

If you see "LangSmith initialization failed" in the logs:

1. Check your `LANGCHAIN_API_KEY` is correct
2. Ensure you have internet connectivity
3. Verify your LangSmith account is active

The system will continue to work without tracing if LangSmith setup fails.
