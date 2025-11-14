import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Configuration
load_dotenv()

DOCUMENT_PATH = "data/faq_document.txt"
PERSIST_DIRECTORY = "./chroma_db"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

def load_document(file_path):
    """Load the FAQ document from the specified path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at {file_path}")
    
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"Loaded document: {file_path}")
    return documents

def chunk_document(documents, chunk_size, chunk_overlap):
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

def create_embeddings():
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
    
    print(f"Initialized embeddings with model: {model}")
    return embeddings

def build_vector_store(chunks, embeddings, persist_dir):
    """Create and persist Chroma vector store from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    print(f"Vector store created and persisted to {persist_dir}")
    return vectorstore

def print_sample_chunks(chunks, n=3):
    """Display sample chunks to verify chunking quality."""
    print(f"\n--- Sample Chunks (showing first {n}) ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Preview: {chunk.page_content[:200]}...")
    print("\n--- End of samples ---\n")

def main():
    """Main pipeline to build the vector index."""
    try:
        print("Starting index building process...\n")
        
        # Load document
        documents = load_document(DOCUMENT_PATH)
        
        # Chunk the document
        chunks = chunk_document(documents, CHUNK_SIZE, CHUNK_OVERLAP)
        
        # Verify we have enough chunks
        if len(chunks) < 20:
            print(f"Warning: Only {len(chunks)} chunks created. Consider adjusting chunk size or adding more content.")
        
        # Show sample chunks
        print_sample_chunks(chunks)
        
        # Create embeddings
        embeddings = create_embeddings()
        
        # Build vector store
        vectorstore = build_vector_store(chunks, embeddings, PERSIST_DIRECTORY)
        
        print("✓ Index building complete!")
        print(f"✓ Total chunks indexed: {len(chunks)}")
        print(f"✓ Vector store saved to: {PERSIST_DIRECTORY}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the FAQ document exists at the specified path.")
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("Check your .env file and ensure all required variables are set.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
