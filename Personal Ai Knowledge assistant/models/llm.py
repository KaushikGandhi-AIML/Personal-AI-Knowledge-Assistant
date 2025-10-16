"""
LLM integration for Personal AI Knowledge Assistant
Following LegalBot pattern with Groq API integration
"""

import os
import json
import logging
from typing import List, Dict, Optional, Generator
from groq import Groq
import openai
from config.config import Config

class PersonalLLM:
    """LLM wrapper for personal knowledge assistant with Groq integration"""
    
    def __init__(self, model_name: str = None):
        """Initialize LLM with Groq API"""
        self.model_name = model_name or Config.DEFAULT_MODEL
        self.temperature = Config.TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        self.logger = logging.getLogger(__name__)
        
        # Initialize Groq client
        try:
            if not Config.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
            self.logger.info(f"Groq client initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {e}")
            raise
        
        # Optional OpenAI fallback
        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.openai_available = True
            self.logger.info("OpenAI fallback available")
        else:
            self.openai_available = False
    
    def generate_response(self, 
                         query: str, 
                         context_documents: List[Dict],
                         conversation_history: List[Dict] = None) -> str:
        """Generate response using RAG context"""
        try:
            # Build prompt with personal context
            prompt = self._build_personal_prompt(query, context_documents, conversation_history)
            
            # Generate response using Groq
            response = self._call_groq_api(prompt)
            
            self.logger.info("Response generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}")
            if self.openai_available:
                return self._fallback_to_openai(prompt)
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def generate_streaming_response(self, 
                                  query: str, 
                                  context_documents: List[Dict],
                                  conversation_history: List[Dict] = None) -> Generator[str, None, None]:
        """Generate streaming response for real-time chat"""
        try:
            prompt = self._build_personal_prompt(query, context_documents, conversation_history)
            
            # Stream response from Groq
            for chunk in self._call_groq_api_streaming(prompt):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Failed to generate streaming response: {e}")
            yield "I apologize, but I'm having trouble generating a response right now."
    
    def _build_personal_prompt(self, 
                              query: str, 
                              context_documents: List[Dict],
                              conversation_history: List[Dict] = None) -> str:
        """Build prompt optimized for personal knowledge assistant"""
        
        # Base system prompt for personal assistant
        system_prompt = """You are a Personal AI Knowledge Assistant. Your role is to help the user access and understand their personal documents, notes, emails, and learning materials.

Key guidelines:
1. Be conversational and helpful, as if you're a knowledgeable friend
2. Always cite the specific documents you're referencing
3. If information comes from multiple sources, mention them all
4. If you can't find relevant information in the provided context, say so clearly
5. Be respectful of personal information and maintain privacy
6. Provide actionable insights when possible
7. If asked about topics not in the context, politely redirect to your knowledge base

Context Documents:
"""
        
        # Add context documents with metadata
        context_text = ""
        for i, doc in enumerate(context_documents, 1):
            context_text += f"\nDocument {i}:\n"
            context_text += f"Source: {doc['metadata'].get('source', 'Unknown')}\n"
            context_text += f"Category: {doc['metadata'].get('category', 'Unknown')}\n"
            context_text += f"Relevance: {doc['similarity']:.2f}\n"
            context_text += f"Content: {doc['text'][:500]}...\n"
            context_text += "---\n"
        
        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages
                history_text += f"{msg['role']}: {msg['content']}\n"
        
        # Combine all parts
        full_prompt = f"{system_prompt}{context_text}{history_text}\n\nUser Query: {query}\n\nResponse:"
        
        return full_prompt
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API for response generation"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Groq API call failed: {e}")
            raise
    
    def _call_groq_api_streaming(self, prompt: str) -> Generator[str, None, None]:
        """Call Groq API for streaming response"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Groq streaming API call failed: {e}")
            yield "Error generating response."
    
    def _fallback_to_openai(self, prompt: str) -> str:
        """Fallback to OpenAI if Groq fails"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.logger.info("Used OpenAI fallback successfully")
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI fallback also failed: {e}")
            return "I apologize, but I'm unable to generate a response at this time."
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'provider': 'Groq',
            'fallback_available': self.openai_available
        }
    
    def update_model_settings(self, 
                            model_name: str = None,
                            temperature: float = None,
                            max_tokens: int = None):
        """Update model settings"""
        if model_name:
            self.model_name = model_name
        if temperature is not None:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens
            
        self.logger.info(f"Model settings updated: {self.get_model_info()}")
