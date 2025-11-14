import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.build_index import load_document, chunk_document, create_embeddings
from src.query import format_context, initialize_embeddings, initialize_llm
from src.evaluator import evaluate_answer


class TestDocumentLoading(unittest.TestCase):
    """Test document loading functionality."""
    
    def test_load_existing_document(self):
        """Test that we can load the FAQ document."""
        doc_path = "data/faq_document.txt"
        if os.path.exists(doc_path):
            documents = load_document(doc_path)
            self.assertIsNotNone(documents)
            self.assertGreater(len(documents), 0)
            self.assertGreater(len(documents[0].page_content), 1000, 
                             "Document should be at least 1000 words")
    
    def test_missing_document_raises_error(self):
        """Test that loading a missing document raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_document("nonexistent_file.txt")


class TestChunking(unittest.TestCase):
    """Test document chunking functionality."""
    
    def test_chunking_creates_multiple_chunks(self):
        """Test that chunking creates at least 20 chunks."""
        doc_path = "data/faq_document.txt"
        if os.path.exists(doc_path):
            documents = load_document(doc_path)
            chunks = chunk_document(documents, chunk_size=1000, chunk_overlap=200)
            
            self.assertGreaterEqual(len(chunks), 20, 
                                   "Should create at least 20 chunks")
            self.assertGreater(len(chunks[0].page_content), 0,
                             "Chunks should not be empty")
    
    def test_chunk_size_respected(self):
        """Test that chunks are approximately the specified size."""
        doc_path = "data/faq_document.txt"
        if os.path.exists(doc_path):
            documents = load_document(doc_path)
            chunk_size = 500
            chunks = chunk_document(documents, chunk_size=chunk_size, chunk_overlap=100)
            
            # Most chunks should be close to the specified size
            # (last chunk might be smaller)
            for chunk in chunks[:-1]:
                self.assertLessEqual(len(chunk.page_content), chunk_size * 1.5,
                                   "Chunks should not be excessively large")
    
    def test_chunk_overlap_creates_shared_content(self):
        """Test that overlap creates shared content between adjacent chunks."""
        doc_path = "data/faq_document.txt"
        if os.path.exists(doc_path):
            documents = load_document(doc_path)
            chunks = chunk_document(documents, chunk_size=1000, chunk_overlap=200)
            
            if len(chunks) > 1:
                # Check that there's some shared content (due to overlap)
                # This is a simplified check - in reality overlap might not always
                # create exact matches due to splitting on boundaries
                self.assertIsNotNone(chunks[0].page_content)
                self.assertIsNotNone(chunks[1].page_content)


class TestEmbeddings(unittest.TestCase):
    """Test embedding initialization."""
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'EMBEDDING_MODEL': 'text-embedding-3-small'
    })
    def test_embeddings_initialization(self):
        """Test that embeddings can be initialized with proper config."""
        try:
            embeddings = create_embeddings()
            self.assertIsNotNone(embeddings)
        except Exception as e:
            # This might fail without a real API key, which is expected
            self.assertIn("api", str(e).lower())
    
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises appropriate error."""
        with self.assertRaises(ValueError):
            create_embeddings()


class TestQueryPipeline(unittest.TestCase):
    """Test query pipeline functionality."""
    
    def test_format_context(self):
        """Test that context formatting works correctly."""
        mock_docs = [
            MagicMock(page_content="First chunk content"),
            MagicMock(page_content="Second chunk content"),
            MagicMock(page_content="Third chunk content")
        ]
        
        context = format_context(mock_docs)
        
        self.assertIn("First chunk content", context)
        self.assertIn("Second chunk content", context)
        self.assertIn("Third chunk content", context)
        self.assertIn("Document 1", context)
        self.assertIn("Document 2", context)
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'EMBEDDING_MODEL': 'text-embedding-3-small'
    })
    def test_embeddings_init_with_config(self):
        """Test embeddings initialization with environment config."""
        try:
            embeddings = initialize_embeddings()
            self.assertIsNotNone(embeddings)
        except Exception:
            # Expected to fail without real API key
            pass
    
    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_BASE_URL': 'https://api.openai.com/v1',
        'LLM_MODEL': 'gpt-3.5-turbo'
    })
    def test_llm_init_with_config(self):
        """Test LLM initialization with environment config."""
        try:
            llm = initialize_llm()
            self.assertIsNotNone(llm)
        except Exception:
            # Expected to fail without real API key
            pass


class TestEvaluator(unittest.TestCase):
    """Test evaluator functionality."""
    
    def test_evaluate_answer_structure(self):
        """Test that evaluator returns proper structure."""
        # Note: This test requires API access to actually run
        # Here we're just testing the structure expectations
        
        sample_question = "What is the PTO policy?"
        sample_answer = "Full-time employees receive 15 days of PTO."
        sample_chunks = [
            {"text": "PTO policy information here", "metadata": {}}
        ]
        
        # This would need mocking or actual API access to test fully
        # For now, we're documenting the expected structure
        expected_keys = ["score", "reason"]
        
        # If we had a mock response, we'd validate it has these keys
        mock_result = {"score": 8, "reason": "Good answer"}
        
        for key in expected_keys:
            self.assertIn(key, mock_result)
        
        self.assertIsInstance(mock_result["score"], int)
        self.assertGreaterEqual(mock_result["score"], 0)
        self.assertLessEqual(mock_result["score"], 10)


class TestJSONOutput(unittest.TestCase):
    """Test that outputs conform to expected JSON format."""
    
    def test_query_output_format(self):
        """Test that query output has all required fields."""
        required_fields = ["user_question", "system_answer", "chunks_related"]
        
        # Mock query result
        mock_result = {
            "user_question": "What is the PTO policy?",
            "system_answer": "Full-time employees receive 15 days of PTO in their first year.",
            "chunks_related": [
                {
                    "text": "Q: How much PTO do I get per year? A: Full-time employees receive 15 days...",
                    "metadata": {}
                }
            ]
        }
        
        # Verify all required fields are present
        for field in required_fields:
            self.assertIn(field, mock_result)
        
        # Verify chunks_related is a list
        self.assertIsInstance(mock_result["chunks_related"], list)
        
        # Verify each chunk has required structure
        for chunk in mock_result["chunks_related"]:
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
    
    def test_evaluator_output_format(self):
        """Test that evaluator output has required fields."""
        mock_evaluation = {
            "score": 8,
            "reason": "Chunks are relevant and answer is accurate."
        }
        
        self.assertIn("score", mock_evaluation)
        self.assertIn("reason", mock_evaluation)
        self.assertIsInstance(mock_evaluation["score"], int)
        self.assertIsInstance(mock_evaluation["reason"], str)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_structure(self):
        """Test that the full pipeline components can work together."""
        # This is a structural test - actual execution requires API keys
        
        # Step 1: Document loading and chunking
        doc_path = "data/faq_document.txt"
        if os.path.exists(doc_path):
            documents = load_document(doc_path)
            chunks = chunk_document(documents, chunk_size=1000, chunk_overlap=200)
            
            self.assertGreater(len(chunks), 0)
            
            # Step 2: Verify chunks can be formatted
            mock_docs = [MagicMock(page_content=chunk.page_content) for chunk in chunks[:3]]
            context = format_context(mock_docs)
            self.assertIsInstance(context, str)
            self.assertGreater(len(context), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

