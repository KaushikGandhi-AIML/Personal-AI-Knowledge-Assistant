"""
Document processing utilities for Personal AI Knowledge Assistant
Handles various file formats and document types
"""

import os
import mimetypes
from typing import List, Dict, Optional
import logging

from config.config import Config
from utils.logger import get_logger

class DocumentProcessor:
    """Process various document formats for personal knowledge base"""
    
    def __init__(self):
        """Initialize document processor"""
        self.logger = get_logger(__name__)
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
        
    def process_file(self, file_path: str, category: str = "other") -> Optional[Dict]:
        """Process a single file and extract content"""
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                return None
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > Config.MAX_FILE_SIZE:
                self.logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return None
            
            # Get file extension
            _, ext = os.path.splitext(file_path.lower())
            if ext not in self.supported_extensions:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None
            
            # Process based on file type
            content = self._extract_content(file_path, ext)
            if not content:
                return None
            
            # Create document metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'file_size': file_size,
                'file_type': ext,
                'category': category,
                'processed_at': os.path.getctime(file_path)
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process file {file_path}: {e}")
            return None
    
    def _extract_content(self, file_path: str, extension: str) -> Optional[str]:
        """Extract content based on file extension"""
        try:
            if extension == '.txt':
                return self._extract_txt(file_path)
            elif extension == '.md':
                return self._extract_md(file_path)
            elif extension == '.pdf':
                return self._extract_pdf(file_path)
            elif extension in ['.docx', '.doc']:
                return self._extract_docx(file_path)
            elif extension == '.eml':
                return self._extract_eml(file_path)
            elif extension == '.mbox':
                return self._extract_mbox(file_path)
            elif extension == '.csv':
                return self._extract_csv(file_path)
            elif extension == '.json':
                return self._extract_json(file_path)
            else:
                return self._extract_txt(file_path)  # Default to text
                
        except Exception as e:
            self.logger.error(f"Failed to extract content from {file_path}: {e}")
            return None
    
    def _extract_txt(self, file_path: str) -> str:
        """Extract content from text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _extract_md(self, file_path: str) -> str:
        """Extract content from Markdown files"""
        return self._extract_txt(file_path)  # Same as text for now
    
    def _extract_pdf(self, file_path: str) -> Optional[str]:
        """Extract content from PDF files"""
        try:
            import PyPDF2
            
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
            
            return content.strip()
            
        except ImportError:
            self.logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract PDF content: {e}")
            return None
    
    def _extract_docx(self, file_path: str) -> Optional[str]:
        """Extract content from Word documents"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            content = ""
            
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            return content.strip()
            
        except ImportError:
            self.logger.error("python-docx not installed. Install with: pip install python-docx")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract DOCX content: {e}")
            return None
    
    def _extract_eml(self, file_path: str) -> Optional[str]:
        """Extract content from EML email files"""
        try:
            import email
            
            with open(file_path, 'rb') as f:
                msg = email.message_from_bytes(f.read())
            
            content = ""
            content += f"From: {msg.get('From', '')}\n"
            content += f"To: {msg.get('To', '')}\n"
            content += f"Subject: {msg.get('Subject', '')}\n"
            content += f"Date: {msg.get('Date', '')}\n\n"
            
            # Extract body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                content += msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return content
            
        except ImportError:
            self.logger.error("email module not available")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract EML content: {e}")
            return None
    
    def _extract_mbox(self, file_path: str) -> Optional[str]:
        """Extract content from MBOX files"""
        try:
            import mailbox
            
            mbox = mailbox.mbox(file_path)
            content = ""
            
            for message in mbox:
                content += f"From: {message.get('From', '')}\n"
                content += f"To: {message.get('To', '')}\n"
                content += f"Subject: {message.get('Subject', '')}\n"
                content += f"Date: {message.get('Date', '')}\n\n"
                
                # Extract body
                if message.is_multipart():
                    for part in message.walk():
                        if part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                content += payload.decode('utf-8', errors='ignore')
                else:
                    payload = message.get_payload(decode=True)
                    if payload:
                        content += payload.decode('utf-8', errors='ignore')
                
                content += "\n---\n"
            
            return content
            
        except ImportError:
            self.logger.error("mailbox module not available")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract MBOX content: {e}")
            return None
    
    def _extract_csv(self, file_path: str) -> Optional[str]:
        """Extract content from CSV files"""
        try:
            import csv
            
            content = ""
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    content += ", ".join(row) + "\n"
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to extract CSV content: {e}")
            return None
    
    def _extract_json(self, file_path: str) -> Optional[str]:
        """Extract content from JSON files"""
        try:
            import json
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert JSON to readable text
            return json.dumps(data, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to extract JSON content: {e}")
            return None
    
    def process_directory(self, directory_path: str, category: str = "other") -> List[Dict]:
        """Process all supported files in a directory"""
        processed_docs = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    doc = self.process_file(file_path, category)
                    if doc:
                        processed_docs.append(doc)
            
            self.logger.info(f"Processed {len(processed_docs)} documents from {directory_path}")
            return processed_docs
            
        except Exception as e:
            self.logger.error(f"Failed to process directory {directory_path}: {e}")
            return []
    
    def get_supported_formats(self) -> Dict[str, str]:
        """Get list of supported file formats"""
        format_descriptions = {
            '.txt': 'Plain Text Files',
            '.md': 'Markdown Files',
            '.pdf': 'PDF Documents',
            '.docx': 'Microsoft Word Documents',
            '.doc': 'Microsoft Word Documents (Legacy)',
            '.rtf': 'Rich Text Format',
            '.eml': 'Email Files',
            '.mbox': 'Mailbox Files',
            '.csv': 'Comma Separated Values',
            '.json': 'JSON Data Files'
        }
        
        return {ext: format_descriptions.get(ext, 'Unknown Format') 
                for ext in self.supported_extensions}
