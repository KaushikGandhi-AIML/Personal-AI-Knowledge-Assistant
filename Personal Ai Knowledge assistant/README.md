# Personal AI Knowledge Assistant 🧠

An AI-powered personal knowledge management system that provides conversational access to your entire personal document collection using Retrieval-Augmented Generation (RAG).

## 🎯 Features

- **Document Ingestion**: Support for multiple file formats (PDF, DOCX, TXT, MD, EML, CSV, JSON)
- **Real-time Processing**: Automatic file monitoring and ingestion
- **Smart Categorization**: Automatic categorization of documents (emails, notes, learning materials, etc.)
- **RAG-powered Chat**: Conversational interface powered by Groq LLM
- **Privacy-focused**: Local embeddings with secure document handling
- **Source Attribution**: Always shows which documents informed responses

## 🏗️ Architecture

Following the proven LegalBot pattern with enhanced personal document processing:

```
├── config/           # Configuration management
├── data/            # Document storage
├── models/          # Embedding and LLM logic
├── utils/           # RAG utilities, logging, processing
├── app.py           # Main Streamlit application
└── requirements.txt # Dependencies
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone or download the project
cd "Personal Ai Knowledge assistant"

# Run setup script
python setup.py

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit the `.env` file with your API keys:

```env
# Get your Groq API key from: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Optional: OpenAI fallback
OPENAI_API_KEY=your_openai_api_key_here

# Change this for security!
ENCRYPTION_KEY=your_secure_encryption_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 📁 Document Categories

The system automatically categorizes documents:

- **📧 Emails**: Email communications (.eml, .mbox)
- **📝 Notes**: Personal notes and memos (.md, .txt)
- **📚 Learning**: Learning materials and study notes
- **🎓 Education**: Educational documents and certificates
- **💼 Work**: Work-related documents
- **👤 Personal**: Personal documents
- **📄 Other**: Miscellaneous documents

## 🔧 Supported File Formats

| Format | Description | Extensions |
|--------|-------------|------------|
| Text Files | Plain text documents | `.txt`, `.md` |
| PDF Documents | Portable Document Format | `.pdf` |
| Word Documents | Microsoft Word files | `.docx`, `.doc` |
| Email Files | Email messages | `.eml`, `.mbox` |
| Data Files | Structured data | `.csv`, `.json` |

## 🎨 Usage Examples

### Upload Documents
1. Use the sidebar file uploader to add documents
2. Documents are automatically categorized and processed
3. View statistics in the sidebar

### Chat with Your Knowledge Base
- Ask questions about your documents
- Get responses with source citations
- Maintain conversation context

### Example Queries
- "What did I learn about machine learning?"
- "Find emails from last month about the project"
- "What are my notes on Python programming?"
- "Summarize my work documents from this week"

## ⚙️ Configuration Options

### Model Settings
- **Response Creativity**: Adjust temperature (0.1-1.0)
- **Search Results**: Number of documents to consider (1-10)

### Document Processing
- **Chunk Size**: Size of document chunks (default: 1000)
- **Chunk Overlap**: Overlap between chunks (default: 200)
- **Similarity Threshold**: Minimum similarity for results (default: 0.7)

## 🔒 Privacy & Security

- **Local Processing**: All embeddings generated locally
- **No Data Sharing**: Documents never leave your system
- **Encryption**: Optional encryption for sensitive data
- **Audit Logging**: Track all queries and document access

## 🛠️ Advanced Features

### File Monitoring
The system can monitor directories for new files:

```python
from utils.file_monitor import PersonalFileMonitor

monitor = PersonalFileMonitor(
    watch_directories=["/path/to/documents"],
    document_processor=document_processor,
    embeddings_model=embeddings_model,
    rag_utils=rag_utils
)

monitor.start_monitoring()
```

### Custom Categories
Add custom document categories in `config/config.py`:

```python
DOCUMENT_CATEGORIES = {
    'research': 'Research Papers',
    'meetings': 'Meeting Notes',
    'projects': 'Project Documents',
    # Add your categories here
}
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your Groq API key is correctly set in `.env`
   - Get your key from: https://console.groq.com/

2. **File Processing Errors**
   - Check file size limits (max 10MB)
   - Ensure file format is supported
   - Check file permissions

3. **Model Loading Issues**
   - Ensure internet connection for model downloads
   - Check available disk space

### Logs
Check the `logs/personal_assistant.log` file for detailed error information.

## 📈 Performance Tips

1. **Batch Processing**: Upload multiple files at once for efficiency
2. **Regular Cleanup**: Use the cleanup utilities for old embeddings
3. **Category Organization**: Organize documents by category for better retrieval

## 🤝 Contributing

This project follows the LegalBot architecture pattern. When extending:

1. Maintain the modular structure
2. Follow the existing logging patterns
3. Add proper error handling
4. Update documentation

## 📄 License

This project is for personal use. Ensure compliance with API terms of service for Groq and OpenAI.

## 🙏 Acknowledgments

- Built following the LegalBot RAG architecture pattern
- Powered by Groq's fast LLM API
- Uses Sentence Transformers for embeddings
- Streamlit for the user interface

---

**Happy Knowledge Management! 🎉**
