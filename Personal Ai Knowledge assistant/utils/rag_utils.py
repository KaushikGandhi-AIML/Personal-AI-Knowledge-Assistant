"""
RAG utilities for Personal AI Knowledge Assistant
Following LegalBot pattern with personal document optimization
"""

import os
import json
import hashlib
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

from config.config import Config
from utils.logger import get_logger

class PersonalRAGUtils:
    """RAG utilities optimized for personal document retrieval"""
    
    def __init__(self):
        """Initialize RAG utilities"""
        self.logger = get_logger(__name__)
        self.document_index = {}
        self.category_index = {}
        
    def chunk_personal_document(self, 
                              content: str, 
                              filename: str,
                              category: str = "other") -> List[Dict]:
        """Chunk personal documents with context preservation"""
        try:
            # Different chunking strategies based on document type
            if category == "emails":
                return self._chunk_emails(content, filename)
            elif category == "notes":
                return self._chunk_notes(content, filename)
            elif category == "learning":
                return self._chunk_learning_materials(content, filename)
            else:
                return self._chunk_general_document(content, filename, category)
                
        except Exception as e:
            self.logger.error(f"Failed to chunk document {filename}: {e}")
            return []
    
    def _chunk_emails(self, content: str, filename: str) -> List[Dict]:
        """Specialized chunking for email content"""
        chunks = []
        
        # Split by email boundaries if multiple emails
        emails = content.split('\n---\n') if '---' in content else [content]
        
        for i, email in enumerate(emails):
            if not email.strip():
                continue
                
            # Extract email metadata
            lines = email.strip().split('\n')
            subject = ""
            sender = ""
            date = ""
            body = ""
            
            # Parse email headers
            in_body = False
            for line in lines:
                if line.startswith('Subject:'):
                    subject = line.replace('Subject:', '').strip()
                elif line.startswith('From:'):
                    sender = line.replace('From:', '').strip()
                elif line.startswith('Date:'):
                    date = line.replace('Date:', '').strip()
                elif line.strip() == '' and not in_body:
                    in_body = True
                    continue
                elif in_body:
                    body += line + '\n'
            
            # Create chunks preserving email context
            chunk_size = Config.CHUNK_SIZE
            overlap = Config.CHUNK_OVERLAP
            
            if len(body) <= chunk_size:
                chunks.append({
                    'content': body.strip(),
                    'metadata': {
                        'filename': filename,
                        'chunk_index': i,
                        'category': 'emails',
                        'subject': subject,
                        'sender': sender,
                        'date': date,
                        'chunk_type': 'complete_email'
                    }
                })
            else:
                # Split long emails into chunks
                start = 0
                chunk_idx = 0
                while start < len(body):
                    end = start + chunk_size
                    chunk_content = body[start:end]
                    
                    chunks.append({
                        'content': chunk_content,
                        'metadata': {
                            'filename': filename,
                            'chunk_index': f"{i}_{chunk_idx}",
                            'category': 'emails',
                            'subject': subject,
                            'sender': sender,
                            'date': date,
                            'chunk_type': 'email_fragment'
                        }
                    })
                    
                    start = end - overlap
                    chunk_idx += 1
        
        return chunks
    
    def _chunk_notes(self, content: str, filename: str) -> List[Dict]:
        """Specialized chunking for personal notes"""
        chunks = []
        
        # Split by natural breaks (double newlines, bullet points, etc.)
        sections = content.split('\n\n')
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Preserve note structure
            chunk_size = Config.CHUNK_SIZE
            if len(section) <= chunk_size:
                chunks.append({
                    'content': section.strip(),
                    'metadata': {
                        'filename': filename,
                        'chunk_index': i,
                        'category': 'notes',
                        'chunk_type': 'note_section'
                    }
                })
            else:
                # Further split large sections
                start = 0
                chunk_idx = 0
                while start < len(section):
                    end = start + chunk_size
                    chunk_content = section[start:end]
                    
                    chunks.append({
                        'content': chunk_content,
                        'metadata': {
                            'filename': filename,
                            'chunk_index': f"{i}_{chunk_idx}",
                            'category': 'notes',
                            'chunk_type': 'note_fragment'
                        }
                    })
                    
                    start = end - Config.CHUNK_OVERLAP
                    chunk_idx += 1
        
        return chunks
    
    def _chunk_learning_materials(self, content: str, filename: str) -> List[Dict]:
        """Specialized chunking for learning materials"""
        chunks = []
        
        # Try to identify sections (headers, chapters, etc.)
        lines = content.split('\n')
        current_section = ""
        section_title = ""
        
        for line in lines:
            # Detect headers (lines that are short and all caps or start with #)
            if (len(line) < 100 and 
                (line.isupper() or line.startswith('#') or line.startswith('Chapter'))):
                # Save previous section
                if current_section.strip():
                    chunks.append({
                        'content': current_section.strip(),
                        'metadata': {
                            'filename': filename,
                            'category': 'learning',
                            'section_title': section_title,
                            'chunk_type': 'learning_section'
                        }
                    })
                
                # Start new section
                section_title = line.strip()
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Add final section
        if current_section.strip():
            chunks.append({
                'content': current_section.strip(),
                'metadata': {
                    'filename': filename,
                    'category': 'learning',
                    'section_title': section_title,
                    'chunk_type': 'learning_section'
                }
            })
        
        return chunks
    
    def _chunk_general_document(self, content: str, filename: str, category: str) -> List[Dict]:
        """General chunking strategy for other document types"""
        chunks = []
        chunk_size = Config.CHUNK_SIZE
        overlap = Config.CHUNK_OVERLAP
        
        start = 0
        chunk_idx = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                for i in range(min(100, chunk_size)):
                    if content[end - i] in '.!?':
                        end = end - i + 1
                        break
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'filename': filename,
                        'chunk_index': chunk_idx,
                        'category': category,
                        'chunk_type': 'general'
                    }
                })
            
            start = end - overlap
            chunk_idx += 1
        
        return chunks
    
    def build_document_index(self, chunks: List[Dict]) -> Dict:
        """Build searchable index for documents"""
        index = {
            'by_category': {},
            'by_filename': {},
            'by_date': {},
            'total_chunks': len(chunks)
        }
        
        for chunk in chunks:
            metadata = chunk['metadata']
            category = metadata.get('category', 'other')
            filename = metadata.get('filename', 'unknown')
            
            # Index by category
            if category not in index['by_category']:
                index['by_category'][category] = []
            index['by_category'][category].append(chunk)
            
            # Index by filename
            if filename not in index['by_filename']:
                index['by_filename'][filename] = []
            index['by_filename'][filename].append(chunk)
        
        return index
    
    def get_document_stats(self) -> Dict:
        """Get statistics about ingested documents"""
        try:
            # Load existing embeddings to get stats
            from models.embeddings import PersonalEmbeddings
            embeddings_model = PersonalEmbeddings()
            _, _, metadata = embeddings_model.load_embeddings()
            
            stats = {
                'total_documents': len(set(m.get('filename') for m in metadata)),
                'total_chunks': len(metadata),
                'categories': {},
                'file_types': {},
                'last_updated': datetime.now().isoformat()
            }
            
            for meta in metadata:
                category = meta.get('category', 'other')
                filename = meta.get('filename', '')
                file_ext = os.path.splitext(filename)[1].lower()
                
                stats['categories'][category] = stats['categories'].get(category, 0) + 1
                stats['file_types'][file_ext] = stats['file_types'].get(file_ext, 0) + 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get document stats: {e}")
            return {'error': str(e)}
    
    def search_by_category(self, category: str, embeddings_model) -> List[Dict]:
        """Search documents by category"""
        try:
            _, texts, metadata = embeddings_model.load_embeddings()
            
            # Filter by category
            category_docs = []
            for i, meta in enumerate(metadata):
                if meta.get('category') == category:
                    category_docs.append({
                        'text': texts[i],
                        'metadata': meta,
                        'index': i
                    })
            
            return category_docs
            
        except Exception as e:
            self.logger.error(f"Failed to search by category {category}: {e}")
            return []
    
    def cleanup_old_embeddings(self, days_to_keep: int = 30):
        """Clean up old embedding files"""
        try:
            import glob
            import time
            
            embedding_files = glob.glob(os.path.join(Config.EMBEDDINGS_DIR, "embeddings_*.pkl"))
            current_time = time.time()
            cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
            
            removed_count = 0
            for file_path in embedding_files:
                if os.path.getctime(file_path) < cutoff_time:
                    os.remove(file_path)
                    removed_count += 1
            
            self.logger.info(f"Cleaned up {removed_count} old embedding files")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old embeddings: {e}")
            return 0
