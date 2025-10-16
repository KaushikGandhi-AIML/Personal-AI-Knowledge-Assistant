"""
Embedding model for Personal AI Knowledge Assistant
Following LegalBot pattern with personal document optimization
"""

import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import logging

from config.config import Config

class PersonalEmbeddings:
    """Embedding model for personal documents with category awareness"""
    
    def __init__(self, model_name: str = None):
        """Initialize embedding model"""
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = None
        self.embeddings_cache = {}
        self.document_metadata = {}
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the embedding model"""
        try:
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Embedding model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if self.model is None:
            self.load_model()
        
        try:
            # Clean and preprocess text for personal documents
            cleaned_text = self._preprocess_text(text)
            embedding = self.model.encode(cleaned_text)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if self.model is None:
            self.load_model()
        
        try:
            # Preprocess all texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            embeddings = self.model.encode(cleaned_texts)
            return embeddings
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for personal document embedding"""
        if not text:
            return ""
        
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # For personal documents, we want to preserve context
        # Don't over-clean as it might remove important personal context
        return text
    
    def store_embeddings(self, embeddings: np.ndarray, 
                        texts: List[str], 
                        metadata: List[Dict]) -> str:
        """Store embeddings with metadata"""
        try:
            # Create embeddings directory if it doesn't exist
            os.makedirs(Config.EMBEDDINGS_DIR, exist_ok=True)
            
            # Generate unique filename
            timestamp = str(int(time.time()))
            filename = f"embeddings_{timestamp}.pkl"
            filepath = os.path.join(Config.EMBEDDINGS_DIR, filename)
            
            # Store embeddings and metadata
            data = {
                'embeddings': embeddings,
                'texts': texts,
                'metadata': metadata,
                'model_name': self.model_name,
                'timestamp': timestamp
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Embeddings stored to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: str = None) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """Load embeddings from file"""
        try:
            if filepath is None:
                # Load the most recent embeddings file
                embedding_files = [f for f in os.listdir(Config.EMBEDDINGS_DIR) 
                                 if f.startswith('embeddings_') and f.endswith('.pkl')]
                if not embedding_files:
                    return np.array([]), [], []
                
                filepath = os.path.join(Config.EMBEDDINGS_DIR, 
                                      sorted(embedding_files)[-1])
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Embeddings loaded from {filepath}")
            return data['embeddings'], data['texts'], data['metadata']
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return np.array([]), [], []
    
    def search_similar_documents(self, 
                               query: str, 
                               embeddings: np.ndarray,
                               texts: List[str],
                               metadata: List[Dict],
                               top_k: int = None) -> List[Dict]:
        """Search for similar documents using cosine similarity"""
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_embedding.reshape(1, -1), 
                embeddings
            )[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] >= Config.SIMILARITY_THRESHOLD:
                    results.append({
                        'text': texts[idx],
                        'metadata': metadata[idx],
                        'similarity': float(similarities[idx]),
                        'index': int(idx)
                    })
            
            self.logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar documents: {e}")
            return []
    
    def get_document_categories(self) -> Dict[str, int]:
        """Get count of documents by category"""
        try:
            _, _, metadata = self.load_embeddings()
            categories = {}
            
            for meta in metadata:
                category = meta.get('category', 'other')
                categories[category] = categories.get(category, 0) + 1
            
            return categories
        except Exception as e:
            self.logger.error(f"Failed to get document categories: {e}")
            return {}

# Import time for timestamp generation
import time
