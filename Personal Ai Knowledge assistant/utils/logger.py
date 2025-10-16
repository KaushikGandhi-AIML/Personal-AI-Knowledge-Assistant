"""
Logging utility for Personal AI Knowledge Assistant
Following LegalBot pattern with enhanced personal document logging
"""

import logging
import os
from datetime import datetime
from config.config import Config

class PersonalLogger:
    """Enhanced logging system for personal knowledge assistant"""
    
    def __init__(self, name: str = __name__):
        """Initialize logger with personal document context"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        os.makedirs(Config.LOGS_DIR, exist_ok=True)
        
        # Set up logging format
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for persistent logging
        self.file_handler = logging.FileHandler(
            os.path.join(Config.LOGS_DIR, 'personal_assistant.log')
        )
        self.file_handler.setFormatter(self.formatter)
        
        # Console handler for real-time feedback
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(self.file_handler)
            self.logger.addHandler(self.console_handler)
    
    def log_document_ingestion(self, filename: str, category: str, status: str):
        """Log document ingestion events"""
        self.logger.info(f"DOCUMENT_INGESTION - File: {filename}, Category: {category}, Status: {status}")
    
    def log_query(self, query: str, response_time: float, sources_used: int):
        """Log user queries and response metrics"""
        # Truncate long queries for logging
        query_preview = query[:100] + "..." if len(query) > 100 else query
        self.logger.info(f"USER_QUERY - Query: {query_preview}, Response Time: {response_time:.2f}s, Sources: {sources_used}")
    
    def log_rag_performance(self, similarity_scores: list, top_k: int):
        """Log RAG retrieval performance"""
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        self.logger.info(f"RAG_PERFORMANCE - Avg Similarity: {avg_similarity:.3f}, Top K: {top_k}")
    
    def log_error(self, error_type: str, error_message: str, context: str = ""):
        """Log errors with context"""
        self.logger.error(f"ERROR - Type: {error_type}, Message: {error_message}, Context: {context}")
    
    def log_security_event(self, event_type: str, details: str):
        """Log security-related events"""
        self.logger.warning(f"SECURITY_EVENT - Type: {event_type}, Details: {details}")
    
    def log_model_usage(self, model_name: str, tokens_used: int, cost: float = None):
        """Log model usage and costs"""
        cost_info = f", Cost: ${cost:.4f}" if cost else ""
        self.logger.info(f"MODEL_USAGE - Model: {model_name}, Tokens: {tokens_used}{cost_info}")

# Create global logger instance
personal_logger = PersonalLogger("PersonalAssistant")

def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger instance"""
    return PersonalLogger(name).logger
