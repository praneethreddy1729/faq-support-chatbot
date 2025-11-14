import os
import sys
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

# Configuration
load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"
TOP_K = int(os.getenv("TOP_K", 3))

def initialize_embeddings():
    """Initialize OpenAI embeddings with configuration from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Fixed: Changed openai_api_base to base_url
    kwargs = {
        "openai_api_key": api_key,
        "model": model
    }
    
    # Only add base_url if it's provided
    if base_url:
        kwargs["base_url"] = base_url
    
    embeddings = OpenAIEmbeddings(**kwargs)
    
    return embeddings

def initialize_llm():
    """Initialize the LLM for answer generation."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Fixed: Changed openai_api_base to base_url
    kwargs = {
        "openai_api_key": api_key,
        "model_name": model,
        "temperature": 0.7
    }
    
    # Only add base_url if it's provided
    if base_url:
        kwargs["base_url"] = base_url
    
    llm = ChatOpenAI(**kwargs)
    
    return llm

def load_vector_store(embeddings):
    """Load the existing Chroma vector store."""
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector store not found at {PERSIST_DIRECTORY}. "
            "Please run 'python src/build_index.py' first."
        )
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    return vectorstore

def retrieve_relevant_chunks(vectorstore, question, k=TOP_K):
    """Retrieve the most relevant chunks for the given question."""
    # Fixed: Use similarity_search() directly instead of retriever
    # This is more straightforward and works with all LangChain versions
    relevant_docs = vectorstore.similarity_search(question, k=k)
    return relevant_docs

def format_context(documents):
    """Format retrieved documents into a context string."""
    context_parts = []
    for i, doc in enumerate(documents, 1):
        context_parts.append(f"Document {i}:\n{doc.page_content}")
    
    return "\n\n".join(context_parts)

def generate_answer(llm, question, context):
    """Generate an answer using the LLM based on retrieved context."""
    prompt = f"""You are a helpful HR assistant. Answer the user's question based on the provided documentation. Be clear, accurate, and professional.

Documentation:
{context}

User Question: {question}

Answer the question based only on the information provided in the documentation. If the documentation doesn't contain enough information to answer fully, acknowledge that and provide what information is available."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def query_system(question, vectorstore, llm, k=TOP_K):
    """Main query function that orchestrates retrieval and generation."""
    # Retrieve relevant chunks
    relevant_docs = retrieve_relevant_chunks(vectorstore, question, k)
    
    # Format context for the LLM
    context = format_context(relevant_docs)
    
    # Generate answer
    answer = generate_answer(llm, question, context)
    
    # Format chunks for JSON output
    chunks_related = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in relevant_docs
    ]
    
    # Create structured output
    result = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks_related
    }
    
    return result

def main():
    """Main entry point for the query system."""
    try:
        # Check if question was provided
        if len(sys.argv) < 2:
            print("Usage: python src/query.py \"your question here\"")
            print("\nExample: python src/query.py \"What is the PTO policy?\"")
            sys.exit(1)
        
        question = " ".join(sys.argv[1:])
        
        # Initialize components
        print("Loading system...", file=sys.stderr)
        embeddings = initialize_embeddings()
        llm = initialize_llm()
        vectorstore = load_vector_store(embeddings)
        
        # Process query
        print(f"Processing query: {question}\n", file=sys.stderr)
        result = query_system(question, vectorstore, llm)
        
        # Output JSON result to stdout
        print(json.dumps(result, indent=2))
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        print("Check your .env file and ensure all required variables are set.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
