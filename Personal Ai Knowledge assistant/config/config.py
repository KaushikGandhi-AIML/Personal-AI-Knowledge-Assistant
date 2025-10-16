"""
Configuration settings for Personal AI Knowledge Assistant
Following LegalBot pattern with personal document focus
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Personal AI Knowledge Assistant"""
    
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # LLM Configuration
    DEFAULT_MODEL = "llama-3.1-70b-versatile"  # Groq model
    TEMPERATURE = 0.7
    MAX_TOKENS = 2048
    
    # Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Supported File Types
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.docx', '.doc', 
        '.rtf', '.eml', '.mbox', '.csv', '.json'
    }
    
    # Personal Document Categories
    DOCUMENT_CATEGORIES = {
        'emails': 'Email Communications',
        'notes': 'Personal Notes',
        'learning': 'Learning Materials',
        'education': 'Education Documents',
        'work': 'Work Documents',
        'resume': 'Resume & CV Documents',
        'other': 'Other Documents'
    }
    
    # Vector Database Settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.7
    TOP_K_RESULTS = 5
    
    # File Paths
    DATA_DIR = "data"
    EMBEDDINGS_DIR = "embeddings"
    LOGS_DIR = "logs"
    
    # Streamlit Configuration
    PAGE_TITLE = "Personal AI Knowledge Assistant"
    PAGE_ICON = "🧠"
    LAYOUT = "wide"
    
    # Security Settings
    ENABLE_ENCRYPTION = True
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "default_key_change_in_production")
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        if not cls.ENCRYPTION_KEY or cls.ENCRYPTION_KEY == "default_key_change_in_production":
            raise ValueError("ENCRYPTION_KEY not set or using default value. Please set a secure encryption key.")
        
        # Create necessary directories
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)
        
        return True
