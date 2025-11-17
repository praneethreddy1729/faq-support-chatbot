# FAQ Support Chatbot

An intelligent FAQ support chatbot built with Retrieval-Augmented Generation (RAG) that answers questions by retrieving relevant information from HR documentation. The system uses vector similarity search to find relevant content and generates accurate, context-aware responses.

## What It Does

This chatbot handles common HR and platform questions by:
1. Breaking down documentation into searchable chunks
2. Converting text into vector embeddings for semantic search
3. Finding the most relevant content for each question
4. Generating accurate answers using retrieved context
5. Providing transparent output showing which documentation was used

## Prerequisites

- Python 3.8 or higher
- API key from OpenRouter or OpenAI
- Internet connection for API calls

## Installation

### Quick Setup

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will check your Python version, create a virtual environment, install dependencies, and create a `.env` file.

### Manual Setup

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

## Configuration

Edit the `.env` file and add your API credentials:

### For OpenRouter

```
OPENAI_API_KEY=your-openrouter-api-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
```

Get your OpenRouter API key from: https://openrouter.ai/keys

### For OpenAI

```
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
```

Get your OpenAI API key from: https://platform.openai.com/api-keys

### Optional Parameters

You can also customize these settings in `.env`:

- `CHUNK_SIZE` - Size of text chunks (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `TOP_K` - Number of relevant chunks to retrieve (default: 3)

## Usage

### Step 1: Build the Index

Before querying, you need to process the FAQ document and build the vector index:

```bash
python src/build_index.py
```

This will:
- Load the FAQ document from `data/faq_document.txt`
- Split it into chunks
- Generate embeddings for each chunk
- Store them in a vector database at `./chroma_db`

Expected output:
```
Starting index building process...

Loaded document: data/faq_document.txt
Created 23 chunks (size=1000, overlap=200)

--- Sample Chunks (showing first 3) ---
...
✓ Index building complete!
✓ Total chunks indexed: 23
✓ Vector store saved to: ./chroma_db
```

### Step 2: Query the System

Ask questions about the documentation:

```bash
python src/query.py "What is the PTO policy?"
```

The system outputs JSON with:
- `user_question` - The question you asked
- `system_answer` - The generated answer
- `chunks_related` - The documentation chunks used to create the answer

Example output:
```json
{
  "user_question": "What is the PTO policy?",
  "system_answer": "Full-time employees receive 15 days of paid time off...",
  "chunks_related": [
    {
      "text": "Q: How much PTO do I get per year?...",
      "metadata": {"source": "data/faq_document.txt"}
    }
  ]
}
```

### Step 3: Evaluate Answer Quality (Optional)

Score the quality of answers using the evaluator:

```bash
python src/query.py "What is the PTO policy?" | python src/evaluator.py -
```

This adds an evaluation section to the output with a score (0-10) and reasoning:

```json
{
  "user_question": "What is the PTO policy?",
  "system_answer": "...",
  "chunks_related": [...],
  "evaluation": {
    "score": 9,
    "reason": "Chunk Relevance (3/3): All chunks directly address PTO..."
  }
}
```

## Project Structure

```
faq_support_chatbot/
├── data/
│   └── faq_document.txt          # Source FAQ documentation (1000+ words)
├── src/
│   ├── build_index.py            # Index building pipeline
│   ├── query.py                  # Query processing and answer generation
│   └── evaluator.py              # Answer quality evaluation
├── tests/
│   └── test_core.py              # Unit tests for core functionality
├── outputs/
│   └── sample_queries.json       # Example query outputs
├── reports/
│   └── architecture_report.md    # Technical decisions and alternatives
├── chroma_db/                    # Vector database (created after build_index.py)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── setup.sh                      # Automated setup script
└── README.md                     # This file
```

## Technical Overview

### Text Chunking
Uses `RecursiveCharacterTextSplitter` to break documents into 1000-character chunks with 200-character overlap. This preserves context at chunk boundaries while keeping chunks small enough for effective retrieval.

### Embeddings
Generates vector embeddings using OpenAI's `text-embedding-3-small` model. These capture semantic meaning, allowing the system to find relevant content even when questions use different wording than the documentation.

### Vector Database
Stores embeddings in Chroma, a local vector database that enables fast similarity search without external dependencies.

### Retrieval
Uses cosine similarity search to find the top 3 most relevant chunks for each question. This balances providing enough context while avoiding information overload.

### Generation
The LLM receives the question along with relevant chunks and generates an answer based solely on the provided context, reducing hallucination and ensuring answers are grounded in documentation.

## Running Tests

Execute the test suite to verify functionality:

```bash
python tests/test_core.py
```

Tests cover:
- Document loading
- Chunking (verifies 20+ chunks are created)
- Embedding initialization
- Context formatting
- JSON output structure

## Example Queries

See `outputs/sample_queries.json` for complete examples. Here are some questions the system can answer:

- "What is the PTO policy?"
- "How do I submit an expense report?"
- "How does the time tracking feature work?"

## Known Limitations

1. **API Dependency**: Requires an active internet connection and valid API key. Calls to OpenRouter/OpenAI can fail if the service is unavailable.

2. **Token Costs**: Each query uses API tokens for embedding generation and LLM responses. Monitor usage to manage costs.

3. **Context Window**: Limited to the top 3 chunks by default. Complex questions requiring information from many different sections might not get complete answers.

4. **Static Knowledge**: The system only knows what's in the FAQ document. It needs re-indexing after document updates.

5. **No Conversational Memory**: Each query is independent. The system doesn't remember previous questions in a conversation.

6. **Embedding Quality**: Answer quality depends on how well the embedding model captures semantic similarity. Some queries might retrieve less relevant chunks if phrasing differs significantly from documentation.

## Troubleshooting

**Problem**: "Vector store not found" error when running query.py  
**Solution**: Run `python src/build_index.py` first to create the index.

**Problem**: "OPENAI_API_KEY not found" error  
**Solution**: Make sure you've created a `.env` file (copy from `.env.example`) and added your API key.

**Problem**: API authentication errors  
**Solution**: Verify your API key is correct and has available credits. Check that `OPENAI_BASE_URL` matches your provider (OpenRouter vs OpenAI).

**Problem**: Slow responses  
**Solution**: This is normal. Each query requires embedding generation and LLM inference, which can take 3-10 seconds depending on the API provider.
