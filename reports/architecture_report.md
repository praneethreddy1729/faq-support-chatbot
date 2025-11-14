# Architecture Report: FAQ Support Chatbot

This document explains the technical decisions made in building the RAG-based FAQ support chatbot, including what we implemented, what alternatives were considered, and why we chose our approach.

## Document Text Chunking

### What We Did
We implemented Sliding Window Chunking with LangChain's `RecursiveCharacterTextSplitter` with a chunk size of 1000 characters and 200 characters of overlap between consecutive chunks.

### Why This Approach
The recursive splitter tries to break text at natural boundaries (double newlines, single newlines, spaces) rather than cutting mid-word or mid-sentence. This preserves semantic meaning better than fixed-size character splits. The 1000-character chunk size is large enough to capture complete question-answer pairs from our FAQ format, but small enough to stay within embedding model limits and keep retrieval focused.

The 200-character overlap creates shared content between adjacent chunks. This helps when relevant information spans chunk boundaries - if someone asks about PTO and the answer continues across two chunks, the overlap increases the chance that one chunk will contain enough context.

### Alternatives Considered

**Fixed-Size Chunking**: Simply splitting every N characters/N tokens regardless of content structure. This is faster but often breaks in the middle of sentences or ideas, making chunks less coherent.

**Recursive Splitting**: Splitting on sentence boundaries and grouping sentences until reaching a size limit. This works well for narrative text but our FAQ format with Q&A pairs benefits more from preserving the question-answer structure.

### Why We Selected Our Option
For FAQ documentation with clear Q&A structure, the recursive character splitter at 1000 characters hits the sweet spot. It respects document structure (splitting at newlines between Q&A pairs), stays within reasonable size limits, and is simple to configure. The overlap provides insurance against edge cases without much downside.

## Embedding Generation

### What We Did
We generate embeddings using OpenAI's `text-embedding-3-small` model.

### Why This Approach
The `text-embedding-3-small` model provides good semantic understanding at a lower cost than larger embedding models. It creates 1536-dimensional vectors that capture meaning well enough to match questions with relevant documentation, even when wording differs. Using an API-based approach means we don't need to manage model weights locally and benefit from OpenAI's infrastructure.

### Alternatives Considered

**Sentence-Transformers (Local Models)**: Models like `all-MiniLM-L6-v2` can run locally without API calls. This eliminates API costs and latency but requires more setup and hardware. For an assignment or small deployment, local models are viable, but they generally underperform OpenAI's models on semantic understanding.

## Vector Database

### What We Did
We use Chroma as our vector store, persisting the database locally in the `./chroma_db` directory.

### Why This Approach
Chroma is a local vector database that stores embeddings on disk and loads them into memory for fast similarity search. It requires no external services, authentication, or network calls during queries (after initial indexing). For a system with a relatively small knowledge base (under 100 chunks), Chroma's in-memory search is effectively instant.

Chroma integrates directly with LangChain, making the code cleaner. The persistence to disk means we build the index once and reuse it for multiple queries, avoiding repeated API calls for embedding generation.

### Alternatives Considered

**FAISS (Facebook AI Similarity Search)**: A highly optimized library for vector similarity search. Faster than Chroma at scale but requires more manual integration. For our use case (dozens of chunks, not millions), the speed difference is negligible.

### Why We Selected Our Option
Chroma hits the right level of capability for this project. It's simple to set up (just a pip install), works entirely locally, integrates with LangChain out of the box, and handles our scale easily. We don't need cloud infrastructure or complex database management.

## Vector Search Method

### What We Did
We perform similarity search using cosine similarity to find the top 3 most relevant chunks for each query. Chroma uses HNSW as it's default indexing algorithm.

### Why This Approach
Cosine similarity measures the angle between two vectors, which effectively captures semantic similarity regardless of vector magnitude. When a user asks a question, we embed it and find the chunks whose embeddings point in the most similar direction in vector space. This works well for matching questions to relevant documentation even when the exact words differ.

Retrieving k=3 chunks balances context and focus. Three chunks typically provide enough information to answer a question without overwhelming the LLM with too much potentially irrelevant content. Since our chunks are around 1000 characters, three chunks is roughly 3000 characters of context, which fits comfortably in any LLM's context window while staying focused.

### Alternatives Considered

**k-NN with Different k Values**: We could retrieve more or fewer chunks. k=1 risks missing relevant information if the top chunk doesn't fully answer the question. k=5 or higher starts including marginally relevant content that can confuse the LLM or dilute the answer. We tested conceptually and k=3 feels right for FAQ content.

**Hybrid Search (Vector + Keyword)**: Combining semantic vector search with traditional keyword/BM25 search. This can improve results when exact terms matter (like product names or specific policy numbers). We could add this if we noticed retrieval missing obvious keyword matches, but for natural language questions about policies and procedures, pure semantic search performs well.

**Reranking**: After retrieving, say, 10 candidates, using a cross-encoder model to rerank them and select the best 3. This improves precision but adds latency and complexity. For our straightforward FAQ matching, the initial similarity search is accurate enough.

### Why We Selected Our Option
Simple cosine similarity with k=3 gives us relevant chunks efficiently. The approach is fast, conceptually clear, and works well for FAQ-style content where questions and documentation are relatively aligned in structure.

## LLM Selection and Prompt Design

### What We Did
We use GPT-3.5-turbo (or equivalent via OpenRouter) to generate answers. The prompt provides retrieved chunks as context and instructs the model to answer based only on that documentation.

### Why This Approach
GPT-3.5-turbo is fast and cost-effective while producing coherent, accurate answers. By explicitly instructing it to answer only from provided documentation, we reduce hallucination - the model is less likely to make things up and more likely to ground its response in the retrieved chunks.

The prompt structure is straightforward:
1. System context ("You are a helpful HR assistant")
2. Documentation chunks
3. User question
4. Instruction to answer based on the docs

This gives the model everything it needs without overcomplicating the prompt engineering.

### Alternatives Considered

**GPT-4 or GPT-4-turbo**: More capable models with better reasoning and fewer hallucinations. However, they cost significantly more per token. For FAQ support where questions are relatively straightforward, GPT-3.5-turbo handles them fine. We could upgrade if we noticed quality issues.

**Smaller Models (GPT-3.5-turbo-16k or older models)**: Older or smaller variants might be cheaper but potentially less capable. The current GPT-3.5-turbo is well-optimized for cost-performance.

**Open Source Models via OpenRouter**: Models like Llama 2, Mistral, or Mixtral available through OpenRouter. These can be cheaper or even free in some cases. Quality varies - some are competitive with GPT-3.5-turbo, others are less reliable. We specify the model via environment variable, so users can experiment with alternatives easily.

**Zero-Shot vs Few-Shot Prompting**: We use zero-shot (just instructions, no examples in the prompt). We could add few-shot examples showing how to format good answers, but for simple Q&A, the model already knows the pattern.

### Why We Selected Our Option
GPT-3.5-turbo with a clear, grounded prompt gives reliable answers at reasonable cost. The flexibility to switch models via environment variables means we're not locked into one choice and can optimize based on actual usage patterns.

## Query Pipeline Architecture

### What We Did
The query pipeline follows this flow:
1. Accept user question (via command line)
2. Initialize embeddings and LLM with API credentials
3. Load the Chroma vector store from disk
4. Embed the user question
5. Retrieve top k relevant chunks via similarity search
6. Format chunks into a context string
7. Send context + question to LLM with grounding instructions
8. Return structured JSON with question, answer, and related chunks

### Why This Approach
This is the standard RAG architecture: retrieve relevant context, then generate an answer conditioned on that context. Separating retrieval and generation makes the system modular and easier to debug. If answers are wrong, we can check whether the issue is retrieval (wrong chunks) or generation (misinterpreting chunks).

Outputting the related chunks alongside the answer provides transparency. Users (or developers) can verify that the answer is based on appropriate documentation. This builds trust and helps identify when the retrieval isn't working correctly.

### Alternatives Considered

**RetrievalQA Chain**: LangChain provides a pre-built `RetrievalQA` chain that handles retrieval and generation. We could use this instead of building our own pipeline. However, the built-in chain is less flexible for custom output formats (we need specific JSON structure with chunks_related). Building the pipeline manually gives us full control.

**Streaming Responses**: We could stream the LLM's response token-by-token for better perceived responsiveness. This would improve UX for interactive use but complicates the JSON output format. For a CLI tool where the full response is needed anyway, non-streaming is simpler.

**Caching**: We could cache query results so repeated questions return instantly without hitting the API. This would reduce costs and latency for common questions. However, it adds complexity (cache invalidation, storage) and for an assignment/prototype, it's premature optimization.

**Query Expansion**: Rewriting or expanding the user's question before retrieval (e.g., adding synonyms or generating multiple query variations). This can improve recall but risks over-complicating simple questions. For FAQ support, users typically ask clear questions that match documentation phrasing reasonably well.

### Why We Selected Our Option
A straightforward pipeline with explicit steps makes the system easy to understand, debug, and extend. The structured JSON output meets the assignment requirements and provides transparency. This architecture is production-ready without unnecessary complexity.

## User Output Format

### What We Did
The system outputs JSON with three fields:
- `user_question`: The original question
- `system_answer`: The generated answer
- `chunks_related`: Array of objects with chunk text and metadata

### Why This Approach
JSON is machine-readable, making the output easy to parse for downstream systems (logging, analytics, UI integration). Including the user question in the output makes each response self-contained - you can log just the JSON and have the full context of what was asked and answered.

Returning the related chunks enables transparency and debugging. A developer can see exactly what documentation was used. A user interface could show "sources" alongside the answer. If the answer is wrong, checking the chunks immediately reveals whether retrieval or generation failed.

### Alternatives Considered

**Plain Text Response**: Just printing the answer as text. Simpler for quick testing but makes automation harder. Without structured output, parsing the response programmatically is fragile.

**More Detailed Metadata**: We could include chunk scores (similarity ratings), reasoning about why each chunk was selected, or confidence metrics. This adds value for debugging but clutters the output and requires additional computation. The chunks themselves provide enough transparency for most purposes.

**Separate Endpoints**: Instead of one JSON response, separate APIs for "get answer" and "get sources." This is more RESTful but adds complexity for a CLI tool. The combined response is more convenient.

**HTML/Markdown Formatting**: Formatting the answer with rich text (bold, lists, links). This improves readability in a UI but requires parsing and rendering. JSON with plain text is more universal.

### Why We Selected Our Option
The JSON format with question, answer, and chunks is clean, transparent, and flexible. It meets all requirements, enables logging and analysis, and works whether the system is used via CLI, API, or UI.

## Evaluator Agent Design

### What We Did
The optional evaluator accepts a query result JSON and uses an LLM to score the answer quality on a 0-10 scale across four criteria:
1. Chunk relevance (0-3 points)
2. Answer accuracy (0-3 points)
3. Completeness (0-2 points)
4. Clarity (0-2 points)

The evaluator returns a JSON object with a score and a detailed reason explaining the breakdown.

### Why This Approach
Automated evaluation provides a way to monitor system quality without manual review of every answer. By breaking scoring into specific criteria, the evaluation is more structured than just "does this seem right?" The LLM can assess whether chunks are relevant and whether the answer accurately reflects those chunks better than rule-based systems could.

Using the same LLM for evaluation that generated the answer might seem circular, but in practice, the explicit evaluation criteria and lower temperature setting make the evaluator more objective. It's catching obvious errors like irrelevant chunks or hallucinated facts.

### Alternatives Considered

**Manual Evaluation**: Having humans score answers. More reliable but doesn't scale and is time-consuming. The LLM evaluator can process hundreds of queries while humans can sample a few for validation.

**Semantic Similarity Scoring**: Computing embedding similarity between the answer and the chunks as a proxy for groundedness. This is faster and cheaper than an LLM call but less nuanced - high similarity doesn't necessarily mean the answer is good, just that it uses similar words.

**Answer-Question Relevance**: Scoring how well the answer addresses the question, ignoring the chunks. This would catch answers that are off-topic but wouldn't verify groundedness in the documentation.

**Ground Truth Comparison**: If we had human-written "correct" answers for common questions, we could compare generated answers against them. This is the gold standard but requires creating and maintaining a reference set.

**Multiple Criteria Models**: Specialized models for each criterion (one for relevance, one for accuracy, etc.). This could be more precise but adds complexity and multiple API calls.

### Why We Selected Our Option
Using the LLM as an evaluator with structured criteria is a pragmatic middle ground. It's automated, relatively cheap, and provides useful signal about answer quality. The detailed reasoning helps identify patterns in failure modes. While not perfect, it's far better than no evaluation at all.

## Summary of Key Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Chunking | RecursiveCharacterTextSplitter, 1000 chars, 200 overlap | Respects structure, preserves context across boundaries |
| Embeddings | OpenAI text-embedding-3-small | Good quality, reasonable cost, API-based simplicity |
| Vector DB | Chroma (local, persistent) | No external dependencies, fast for small scale, LangChain integration |
| Search | Cosine similarity, k=3 | Simple, effective, balanced context |
| LLM | GPT-3.5-turbo | Cost-effective, reliable, grounded prompting reduces hallucination |
| Output | JSON with question, answer, chunks | Transparent, machine-readable, meets requirements |
| Evaluation | LLM-based scoring with criteria | Automated quality monitoring, detailed feedback |

All choices prioritize simplicity, transparency, and production-readiness while meeting the assignment requirements for a functional RAG-based FAQ system.
