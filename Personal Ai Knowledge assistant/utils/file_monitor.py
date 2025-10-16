"""
File monitoring system for Personal AI Knowledge Assistant
Following LegalBot pattern with real-time document ingestion
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, Callable, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
import logging

from config.config import Config
from utils.logger import get_logger

class PersonalFileHandler(FileSystemEventHandler):
    """File system event handler for personal documents"""
    
    def __init__(self, 
                 document_processor,
                 embeddings_model,
                 rag_utils,
                 category_mapping: Dict[str, str] = None):
        """Initialize file handler"""
        self.document_processor = document_processor
        self.embeddings_model = embeddings_model
        self.rag_utils = rag_utils
        self.category_mapping = category_mapping or {}
        self.logger = get_logger(__name__)
        self.processing_queue = []
        self.processing_lock = threading.Lock()
    
    def on_created(self, event):
        """Handle file creation events"""
        if not event.is_directory:
            self._queue_file_for_processing(event.src_path, "created")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory:
            self._queue_file_for_processing(event.src_path, "modified")
    
    def _queue_file_for_processing(self, file_path: str, event_type: str):
        """Queue file for processing"""
        try:
            # Check if file is supported
            _, ext = os.path.splitext(file_path.lower())
            if ext not in Config.SUPPORTED_EXTENSIONS:
                return
            
            # Determine category based on directory structure
            category = self._determine_category(file_path)
            
            with self.processing_lock:
                self.processing_queue.append({
                    'file_path': file_path,
                    'event_type': event_type,
                    'category': category,
                    'timestamp': time.time()
                })
            
            self.logger.info(f"Queued file for processing: {file_path} ({event_type})")
            
            # Process immediately in a separate thread
            threading.Thread(
                target=self._process_file_async,
                args=(file_path, event_type, category),
                daemon=True
            ).start()
            
        except Exception as e:
            self.logger.error(f"Failed to queue file {file_path}: {e}")
    
    def _determine_category(self, file_path: str) -> str:
        """Determine document category based on file path"""
        try:
            path_parts = Path(file_path).parts
            
            # Check for category keywords in path
            for part in path_parts:
                part_lower = part.lower()
                if 'email' in part_lower or 'mail' in part_lower:
                    return 'emails'
                elif 'note' in part_lower:
                    return 'notes'
                elif 'learn' in part_lower or 'study' in part_lower:
                    return 'learning'
                elif 'work' in part_lower or 'job' in part_lower:
                    return 'work'
                elif 'education' in part_lower or 'school' in part_lower:
                    return 'education'
                elif 'personal' in part_lower:
                    return 'personal'
            
            # Check file extension for hints
            _, ext = os.path.splitext(file_path.lower())
            if ext == '.eml':
                return 'emails'
            elif ext == '.md':
                return 'notes'
            
            return 'other'
            
        except Exception as e:
            self.logger.error(f"Failed to determine category for {file_path}: {e}")
            return 'other'
    
    def _process_file_async(self, file_path: str, event_type: str, category: str):
        """Process file asynchronously"""
        try:
            # Wait a bit to ensure file is fully written
            time.sleep(1)
            
            # Check if file still exists and is accessible
            if not os.path.exists(file_path):
                return
            
            # Process the file
            doc = self.document_processor.process_file(file_path, category)
            if not doc:
                self.logger.warning(f"Failed to process file: {file_path}")
                return
            
            # Add to knowledge base
            success = self._add_to_knowledge_base(doc)
            
            if success:
                self.logger.info(f"Successfully processed {file_path} ({event_type})")
            else:
                self.logger.error(f"Failed to add {file_path} to knowledge base")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
    
    def _add_to_knowledge_base(self, document: Dict) -> bool:
        """Add document to knowledge base"""
        try:
            # Chunk the document
            chunks = self.rag_utils.chunk_personal_document(
                document['content'],
                document['metadata']['filename'],
                document['metadata']['category']
            )
            
            if not chunks:
                return False
            
            # Generate embeddings for chunks
            texts = [chunk['content'] for chunk in chunks]
            metadata = [chunk['metadata'] for chunk in chunks]
            
            embeddings = self.embeddings_model.generate_embeddings_batch(texts)
            
            # Store embeddings
            self.embeddings_model.store_embeddings(embeddings, texts, metadata)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document to knowledge base: {e}")
            return False

class PersonalFileMonitor:
    """File monitoring system for personal documents"""
    
    def __init__(self, 
                 watch_directories: list,
                 document_processor,
                 embeddings_model,
                 rag_utils):
        """Initialize file monitor"""
        self.watch_directories = watch_directories
        self.document_processor = document_processor
        self.embeddings_model = embeddings_model
        self.rag_utils = rag_utils
        self.observer = Observer()
        self.file_handler = None
        self.is_monitoring = False
        self.logger = get_logger(__name__)
    
    def start_monitoring(self):
        """Start monitoring directories"""
        try:
            if self.is_monitoring:
                self.logger.warning("File monitoring already running")
                return
            
            # Initialize file handler
            self.file_handler = PersonalFileHandler(
                self.document_processor,
                self.embeddings_model,
                self.rag_utils
            )
            
            # Add watch directories
            for directory in self.watch_directories:
                if os.path.exists(directory):
                    self.observer.schedule(
                        self.file_handler,
                        directory,
                        recursive=True
                    )
                    self.logger.info(f"Monitoring directory: {directory}")
                else:
                    self.logger.warning(f"Directory not found: {directory}")
            
            # Start observer
            self.observer.start()
            self.is_monitoring = True
            self.logger.info("File monitoring started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start file monitoring: {e}")
            raise
    
    def stop_monitoring(self):
        """Stop monitoring directories"""
        try:
            if not self.is_monitoring:
                return
            
            self.observer.stop()
            self.observer.join()
            self.is_monitoring = False
            self.logger.info("File monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop file monitoring: {e}")
    
    def get_status(self) -> Dict:
        """Get monitoring status"""
        return {
            'is_monitoring': self.is_monitoring,
            'watch_directories': self.watch_directories,
            'queue_size': len(self.file_handler.processing_queue) if self.file_handler else 0
        }
    
    def process_existing_files(self):
        """Process all existing files in watch directories"""
        try:
            processed_count = 0
            
            for directory in self.watch_directories:
                if not os.path.exists(directory):
                    continue
                
                # Process all supported files in directory
                docs = self.document_processor.process_directory(directory)
                
                for doc in docs:
                    # Add to knowledge base
                    chunks = self.rag_utils.chunk_personal_document(
                        doc['content'],
                        doc['metadata']['filename'],
                        doc['metadata']['category']
                    )
                    
                    if chunks:
                        texts = [chunk['content'] for chunk in chunks]
                        metadata = [chunk['metadata'] for chunk in chunks]
                        embeddings = self.embeddings_model.generate_embeddings_batch(texts)
                        self.embeddings_model.store_embeddings(embeddings, texts, metadata)
                        processed_count += 1
            
            self.logger.info(f"Processed {processed_count} existing files")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Failed to process existing files: {e}")
            return 0
