"""
Personal AI Knowledge Assistant - Main Streamlit Application
Following LegalBot pattern with personal document focus
"""

import streamlit as st
import os
import time
from datetime import datetime
from typing import List, Dict

# Import our custom modules
from config.config import Config
from models.embeddings import PersonalEmbeddings
from models.llm import PersonalLLM
from utils.rag_utils import PersonalRAGUtils
from utils.document_processor import DocumentProcessor
from utils.logger import get_logger

# Configure Streamlit page
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

class PersonalKnowledgeAssistant:
    """Main application class for Personal AI Knowledge Assistant"""
    
    def __init__(self):
        """Initialize the application"""
        self.logger = get_logger(__name__)
        self.embeddings_model = None
        self.llm_model = None
        self.rag_utils = PersonalRAGUtils()
        self.document_processor = DocumentProcessor()
        
        # Initialize session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'embeddings_loaded' not in st.session_state:
            st.session_state.embeddings_loaded = False
        if 'document_stats' not in st.session_state:
            st.session_state.document_stats = {}
    
    def initialize_models(self):
        """Initialize embedding and LLM models"""
        try:
            if not st.session_state.embeddings_loaded:
                with st.spinner("Loading AI models..."):
                    self.embeddings_model = PersonalEmbeddings()
                    self.llm_model = PersonalLLM()
                    st.session_state.embeddings_loaded = True
                st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Failed to initialize models: {e}")
            return False
        return True
    
    def render_sidebar(self):
        """Render the sidebar with document management"""
        st.sidebar.title("📚 Document Management")
        
        # Document upload section
        st.sidebar.subheader("📤 Upload Documents")
        uploaded_files = st.sidebar.file_uploader(
            "Choose files to add to your knowledge base",
            type=['txt', 'pdf', 'docx', 'md', 'eml', 'csv', 'json'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, MD, EML, CSV, JSON"
        )
        
        if uploaded_files:
            self.handle_file_uploads(uploaded_files)
        
        # Document categories
        st.sidebar.subheader("📁 Document Categories")
        category = st.sidebar.selectbox(
            "Select category for new documents",
            options=list(Config.DOCUMENT_CATEGORIES.keys()),
            format_func=lambda x: Config.DOCUMENT_CATEGORIES[x]
        )
        
        # Document statistics
        st.sidebar.subheader("📊 Knowledge Base Stats")
        self.display_document_stats()
        
        # Settings
        st.sidebar.subheader("⚙️ Settings")
        self.render_settings()
    
    def handle_file_uploads(self, uploaded_files):
        """Handle uploaded files"""
        if st.sidebar.button("Process Uploaded Files"):
            with st.spinner("Processing documents..."):
                processed_count = 0
                
                for uploaded_file in uploaded_files:
                    try:
                        # Save uploaded file temporarily
                        temp_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process the file
                        doc = self.document_processor.process_file(temp_path)
                        if doc:
                            # Generate embeddings
                            self.add_document_to_knowledge_base(doc)
                            processed_count += 1
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {e}")
                
                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} documents!")
                    st.session_state.document_stats = self.rag_utils.get_document_stats()
                    st.rerun()
    
    def add_document_to_knowledge_base(self, document: Dict):
        """Add document to the knowledge base"""
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
            
            self.logger.log_document_ingestion(
                document['metadata']['filename'],
                document['metadata']['category'],
                "success"
            )
            
            return True
            
        except Exception as e:
            self.logger.log_error("document_ingestion", str(e), document['metadata']['filename'])
            return False
    
    def display_document_stats(self):
        """Display document statistics"""
        try:
            stats = self.rag_utils.get_document_stats()
            
            if stats.get('error'):
                st.error(f"Error loading stats: {stats['error']}")
                return
            
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
                st.metric("Total Chunks", stats.get('total_chunks', 0))
            
            with col2:
                if stats.get('categories'):
                    st.write("**By Category:**")
                    for category, count in stats['categories'].items():
                        st.write(f"• {category}: {count}")
            
            if stats.get('file_types'):
                st.write("**File Types:**")
                for file_type, count in stats['file_types'].items():
                    st.write(f"• {file_type}: {count}")
                    
        except Exception as e:
            st.error(f"Failed to load document stats: {e}")
    
    def render_settings(self):
        """Render settings panel"""
        # Model settings
        temperature = st.sidebar.slider(
            "Response Creativity",
            min_value=0.1,
            max_value=1.0,
            value=Config.TEMPERATURE,
            step=0.1,
            help="Higher values make responses more creative"
        )
        
        max_results = st.sidebar.slider(
            "Max Search Results",
            min_value=1,
            max_value=10,
            value=Config.TOP_K_RESULTS,
            help="Number of documents to consider for responses"
        )
        
        # Update config
        if self.llm_model:
            self.llm_model.update_model_settings(temperature=temperature)
        Config.TOP_K_RESULTS = max_results
        
        # Clear conversation
        if st.sidebar.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("🧠 Personal AI Knowledge Assistant")
        st.markdown("Ask me anything about your personal documents, notes, and learning materials!")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for source in message["sources"]:
                            st.write(f"**{source['metadata'].get('filename', 'Unknown')}**")
                            st.write(f"Category: {source['metadata'].get('category', 'Unknown')}")
                            st.write(f"Relevance: {source['similarity']:.2f}")
                            st.write("---")
        
        # Chat input
        if prompt := st.chat_input("Ask about your documents..."):
            self.handle_user_query(prompt)
    
    def handle_user_query(self, query: str):
        """Handle user query with RAG"""
        try:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching your knowledge base..."):
                    start_time = time.time()
                    
                    # Search for relevant documents
                    _, texts, metadata = self.embeddings_model.load_embeddings()
                    
                    if not texts:
                        response = "I don't have any documents in my knowledge base yet. Please upload some documents first!"
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        return
                    
                    # Find similar documents
                    similar_docs = self.embeddings_model.search_similar_documents(
                        query, 
                        self.embeddings_model.generate_embeddings_batch(texts),
                        texts,
                        metadata
                    )
                    
                    if not similar_docs:
                        response = "I couldn't find relevant information in your documents for this query. Try rephrasing or upload more relevant documents."
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        return
                    
                    # Generate response with context
                    conversation_history = st.session_state.messages[-5:]  # Last 5 messages
                    response = self.llm_model.generate_response(
                        query, 
                        similar_docs, 
                        conversation_history
                    )
                    
                    response_time = time.time() - start_time
                    
                    # Log the query
                    self.logger.log_query(query, response_time, len(similar_docs))
                    
                    # Display response
                    st.markdown(response)
                    
                    # Show sources
                    with st.expander("📚 Sources Used"):
                        for doc in similar_docs:
                            st.write(f"**{doc['metadata'].get('filename', 'Unknown')}**")
                            st.write(f"Category: {doc['metadata'].get('category', 'Unknown')}")
                            st.write(f"Relevance: {doc['similarity']:.2f}")
                            st.write("---")
                    
                    # Add to conversation
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": similar_docs
                    })
                    
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {e}"
            st.error(error_msg)
            self.logger.log_error("query_processing", str(e), query)
    
    def run(self):
        """Run the main application"""
        try:
            # Validate configuration
            Config.validate_config()
            
            # Initialize models
            if not self.initialize_models():
                st.error("Failed to initialize AI models. Please check your API keys.")
                return
            
            # Render interface
            self.render_sidebar()
            self.render_chat_interface()
            
        except Exception as e:
            st.error(f"Application error: {e}")
            self.logger.log_error("application", str(e))

def main():
    """Main entry point"""
    app = PersonalKnowledgeAssistant()
    app.run()

if __name__ == "__main__":
    main()
