import streamlit as st
import os
import sqlite3
import json
import pickle
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import tempfile
import time
import re
import asyncio
import aiohttp
import urllib.parse
from pathlib import Path

# Core ML and NLP libraries
from sentence_transformers import SentenceTransformer
import faiss

# Document processing
import PyPDF2
from io import StringIO
import docx

# Web scraping
import requests
from bs4 import BeautifulSoup

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FAISSVectorStore:
    """FAISS-based vector store - ChromaDB free"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.metadatas = []
        self.ids = []
        self.dimension = None
        
    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents to the vector store"""
        if not texts:
            return
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        if self.index is None:
            # Initialize FAISS index
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
    def similarity_search(self, query: str, k: int = 4) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't search for more than available
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):  # Valid index
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'score': float(scores[0][i]),
                    'id': self.ids[idx]
                })
                
        return results
        
    def save_to_disk(self, path: str):
        """Save the vector store to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, f"{path}.faiss")
                
            # Save documents and metadata
            with open(f"{path}_data.pkl", "wb") as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids,
                    'dimension': self.dimension
                }, f)
        except Exception as e:
            st.error(f"Error saving vector store: {e}")
            
    def load_from_disk(self, path: str) -> bool:
        """Load the vector store from disk"""
        try:
            if os.path.exists(f"{path}.faiss") and os.path.exists(f"{path}_data.pkl"):
                self.index = faiss.read_index(f"{path}.faiss")
                
                with open(f"{path}_data.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
                    self.dimension = data['dimension']
                return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
        return False

class WebLegalSearcher:
    """Web scraper for legal databases"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_legal_databases(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search multiple legal databases"""
        all_results = []
        
        # Search IndianKanoon
        ik_results = self._search_indiankanoon(query, max_results // 2)
        all_results.extend(ik_results)
        
        # Search Google Scholar
        scholar_results = self._search_google_scholar(query, max_results // 2)
        all_results.extend(scholar_results)
        
        return all_results[:max_results]
    
    def _search_indiankanoon(self, query: str, max_results: int = 3) -> List[Dict]:
        """Search IndianKanoon"""
        try:
            search_url = f"https://indiankanoon.org/search/?formInput={urllib.parse.quote(query)}"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find result divs
            result_divs = soup.find_all('div', class_='result')
            if not result_divs:
                result_divs = soup.find_all('div', {'class': re.compile(r'result|search')})
            
            for div in result_divs[:max_results]:
                link = div.find('a', href=True)
                if link and '/doc/' in link.get('href', ''):
                    title = link.get_text(strip=True)
                    if len(title) > 10:
                        case_url = f"https://indiankanoon.org{link['href']}"
                        
                        snippet_div = div.find('div', class_='snippet') or div
                        snippet = snippet_div.get_text(strip=True)[:300]
                        
                        results.append({
                            'title': title,
                            'url': case_url,
                            'snippet': snippet,
                            'source': 'IndianKanoon',
                            'type': 'case'
                        })
            
            return results
        except Exception as e:
            print(f"IndianKanoon search error: {e}")
            return []
    
    def _search_google_scholar(self, query: str, max_results: int = 2) -> List[Dict]:
        """Search Google Scholar"""
        try:
            search_query = f'"{query}" law case court'
            search_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(search_query)}&hl=en"
            
            response = self.session.get(search_url, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            result_divs = soup.find_all('div', {'class': 'gs_r'})[:max_results]
            
            for div in result_divs:
                title_elem = div.find('h3')
                if title_elem:
                    title_link = title_elem.find('a')
                    title = title_elem.get_text(strip=True)
                    url = title_link.get('href') if title_link else None
                    
                    snippet_elem = div.find('div', {'class': 'gs_rs'})
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and len(title) > 10:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet[:300],
                            'source': 'Google Scholar',
                            'type': 'academic'
                        })
            
            return results
        except Exception as e:
            print(f"Google Scholar search error: {e}")
            return []

class DocumentProcessor:
    """Document processor for legal documents"""
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return ""
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to end at a sentence boundary
            if end < len(text):
                for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + chunk_size - overlap, end)
            
        return chunks

class LegalRAGSystem:
    """Main Legal RAG System - FAISS based, error-free"""
    
    def __init__(self):
        self.embedding_model = None
        self.vector_store = None
        self.document_processor = DocumentProcessor()
        self.web_searcher = WebLegalSearcher()
        self.db_path = "legal_knowledge.db"
        self.vector_store_path = "legal_vector_store"
        
        # Initialize database
        self.setup_database()
        
        # API configuration
        self.api_keys = {
            'perplexity': os.getenv('PERPLEXITY_API_KEY'),
            'gemini': os.getenv('GOOGLE_API_KEY'),
            'openai': os.getenv('OPENAI_API_KEY')
        }
        
        # Initialize models
        self.initialize_models()
        
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def initialize_models(self):
        """Initialize embedding model and vector store"""
        try:
            if self.embedding_model is None:
                with st.spinner("Loading AI models..."):
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
            if self.vector_store is None:
                self.vector_store = FAISSVectorStore(self.embedding_model)
                # Try to load existing vector store
                if not self.vector_store.load_from_disk(self.vector_store_path):
                    st.info("🚀 New system initialized. Upload documents to get started!")
                    
        except Exception as e:
            st.error(f"Error initializing models: {e}")
    
    def add_document(self, file_path: str, filename: str) -> str:
        """Add a document to the knowledge base"""
        try:
            # Extract text based on file type
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                text = self.document_processor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                text = self.document_processor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                text = self.document_processor.extract_text_from_txt(file_path)
            else:
                return f"❌ Unsupported file type: {file_extension}"
            
            if not text.strip():
                return "❌ No text content found in document"
            
            # Split into chunks
            chunks = self.document_processor.split_text_into_chunks(text)
            
            # Generate unique IDs
            doc_id = hashlib.md5((filename + str(datetime.now())).encode()).hexdigest()[:12]
            
            # Prepare data for vector store
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{
                'filename': filename,
                'doc_id': doc_id,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'doc_type': 'legal',
                'upload_date': datetime.now().isoformat()
            } for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadatas, chunk_ids)
            
            # Save vector store
            self.vector_store.save_to_disk(self.vector_store_path)
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, filename, content, doc_type, summary)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, filename, text[:1000], 'legal', f"Document with {len(chunks)} chunks"))
            
            conn.commit()
            conn.close()
            
            return f"✅ Successfully added: {filename} ({len(chunks)} chunks, {len(text)} characters)"
            
        except Exception as e:
            return f"❌ Error processing document: {str(e)}"
    
    def search_knowledge_base(self, query: str, k: int = 4) -> Tuple[List[str], List[Dict]]:
        """Search the local knowledge base"""
        try:
            if not self.vector_store or self.vector_store.index is None:
                return [], []
            
            results = self.vector_store.similarity_search(query, k)
            
            contexts = []
            metadatas = []
            
            for result in results:
                contexts.append(result['content'])
                metadatas.append(result['metadata'])
            
            return contexts, metadatas
            
        except Exception as e:
            print(f"Knowledge base search error: {e}")
            return [], []
    
    def query_ai_api(self, prompt: str, provider: str = "auto") -> str:
        """Query AI APIs with fallback support"""
        
        # Auto-select provider based on available API keys
        if provider == "auto":
            if self.api_keys['perplexity']:
                provider = "perplexity"
            elif self.api_keys['gemini']:
                provider = "gemini"
            elif self.api_keys['openai']:
                provider = "openai"
            else:
                return "❌ No API keys configured. Please add your API keys to the .env file."
        
        try:
            if provider == "perplexity" and self.api_keys['perplexity']:
                return self._query_perplexity(prompt)
            elif provider == "gemini" and self.api_keys['gemini']:
                return self._query_gemini(prompt)
            elif provider == "openai" and self.api_keys['openai']:
                return self._query_openai(prompt)
            else:
                return "❌ Selected AI provider not available or API key missing."
                
        except Exception as e:
            # Fallback to basic response
            return f"⚠️ AI service temporarily unavailable. Based on available information: {prompt[:500]}"
    
    def _query_perplexity(self, prompt: str) -> str:
        """Query Perplexity API"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_keys['perplexity'],
                base_url="https://api.perplexity.ai"
            )
            
            response = client.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=3000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Perplexity API error: {e}")
    
    def _query_gemini(self, prompt: str) -> str:
        """Query Google Gemini API"""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_keys['gemini'])
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=3000
                )
            )
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_keys['openai'])
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=3000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def generate_legal_answer(self, question: str, context: str = "", include_web_search: bool = True) -> Dict:
        """Generate comprehensive legal answer"""
        try:
            # Step 1: Search local knowledge base
            local_contexts, local_metadatas = self.search_knowledge_base(question)
            
            # Step 2: Web search (if enabled)
            web_results = []
            if include_web_search:
                try:
                    web_results = self.web_searcher.search_legal_databases(question, 3)
                except Exception as e:
                    print(f"Web search error: {e}")
            
            # Step 3: Combine all information
            all_contexts = local_contexts.copy()
            all_sources = []
            
            # Add local sources
            for metadata in local_metadatas:
                all_sources.append({
                    'type': 'local',
                    'filename': metadata.get('filename', 'Unknown'),
                    'source': 'Local Knowledge Base'
                })
            
            # Add web sources
            for web_result in web_results:
                all_contexts.append(web_result.get('snippet', ''))
                all_sources.append({
                    'type': 'web',
                    'title': web_result.get('title', 'Unknown'),
                    'source': web_result.get('source', 'Web'),
                    'url': web_result.get('url', '')
                })
            
            # Step 4: Generate AI response
            if all_contexts:
                combined_context = "\n\n".join(all_contexts[:8])  # Limit context size
                
                ai_prompt = f"""As an expert legal analyst, provide a comprehensive answer to the following legal question based on the provided context.

Question: {question}

Additional Context: {context}

Legal Sources and Information:
{combined_context}

Instructions:
1. Provide a direct, professional answer to the legal question
2. Reference specific legal principles, cases, or statutes mentioned in the sources
3. Include practical advice and next steps if applicable
4. Highlight any important considerations or potential challenges
5. Structure your response clearly with headings if needed
6. If the sources don't contain sufficient information, acknowledge this

Professional Legal Analysis:"""

                ai_response = self.query_ai_api(ai_prompt)
            else:
                # Fallback when no context is available
                ai_prompt = f"""As an expert legal analyst, provide a comprehensive answer to this legal question:

Question: {question}

Additional Context: {context}

Please provide:
1. A professional legal analysis
2. General legal principles that apply
3. Common considerations for this type of issue
4. Recommended next steps
5. Important disclaimers

Note: This analysis is based on general legal principles. Specific legal advice requires consultation with a qualified attorney.

Professional Legal Analysis:"""

                ai_response = self.query_ai_api(ai_prompt)
            
            # Step 5: Prepare response
            response = {
                'answer': ai_response,
                'sources': all_sources,
                'context_used': len(all_contexts),
                'web_results_found': len(web_results),
                'local_results_found': len(local_contexts),
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 6: Save to chat history
            self.save_chat_history(question, ai_response)
            
            return response
            
        except Exception as e:
            return {
                'answer': f"❌ Error generating response: {str(e)}",
                'sources': [],
                'context_used': 0,
                'web_results_found': 0,
                'local_results_found': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def save_chat_history(self, question: str, answer: str):
        """Save chat interaction to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO chat_history (question, answer)
                VALUES (?, ?)
            ''', (question, answer[:1000]))  # Truncate for storage
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Chat history save error: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM documents')
            doc_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM chat_history')
            chat_count = cursor.fetchone()[0]
            
            conn.close()
            
            vector_count = self.vector_store.index.ntotal if self.vector_store and self.vector_store.index else 0
            
            return {
                'documents': doc_count,
                'chunks': vector_count,
                'chats': chat_count,
                'api_status': {
                    'perplexity': '✅' if self.api_keys['perplexity'] else '❌',
                    'gemini': '✅' if self.api_keys['gemini'] else '❌',
                    'openai': '✅' if self.api_keys['openai'] else '❌'
                }
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Best Legal RAG System",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .answer-container {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sources-container {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Best Legal RAG System</h1>
        <h3>FAISS-Based • Error-Free • Production Ready</h3>
        <p>Advanced Legal Research • Document Processing • AI-Powered Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if 'legal_rag_system' not in st.session_state:
        try:
            st.session_state.legal_rag_system = LegalRAGSystem()
            st.success("✅ System initialized successfully!")
        except Exception as e:
            st.error(f"❌ System initialization error: {e}")
            st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # API Status
        stats = st.session_state.legal_rag_system.get_system_stats()
        
        if 'api_status' in stats:
            st.subheader("🤖 AI Providers")
            for provider, status in stats['api_status'].items():
                st.write(f"**{provider.title()}**: {status}")
        
        st.divider()
        
        # Upload Section
        st.subheader("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose legal documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files"
        )
        
        if uploaded_files:
            if st.button("📤 Process All Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save temporary file
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process document
                    result = st.session_state.legal_rag_system.add_document(temp_path, uploaded_file.name)
                    
                    if "✅" in result:
                        success_count += 1
                        st.success(f"✅ {uploaded_file.name}")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {result}")
                    
                    # Clean up
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                
                status_text.text(f"✅ Complete! {success_count}/{len(uploaded_files)} successful")
                time.sleep(2)
                status_text.empty()
                progress_bar.empty()
                
                st.rerun()
        
        st.divider()
        
        # System Stats
        st.subheader("📊 System Statistics")
        
        if 'error' not in stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('documents', 0))
                st.metric("Chunks", stats.get('chunks', 0))
            with col2:
                st.metric("Chat History", stats.get('chats', 0))
        else:
            st.error(f"Stats error: {stats['error']}")
        
        # Settings
        st.subheader("⚙️ Settings")
        
        ai_provider = st.selectbox(
            "🤖 AI Provider",
            ["auto", "perplexity", "gemini", "openai"],
            help="Select which AI provider to use"
        )
        
        enable_web_search = st.checkbox(
            "🌐 Enable Web Search",
            value=True,
            help="Include real-time web search results"
        )
    
    # Main interface
    st.header("💬 Ask Legal Questions")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display answer
                st.markdown(f"""
                <div class="answer-container">
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            if source.get('type') == 'local':
                                st.write(f"**{i}.** 📄 {source.get('filename', 'Unknown')} (Local Document)")
                            elif source.get('type') == 'web':
                                title = source.get('title', 'Unknown')
                                url = source.get('url', '')
                                if url:
                                    st.write(f"**{i}.** 🌐 [{title}]({url}) ({source.get('source', 'Web')})")
                                else:
                                    st.write(f"**{i}.** 🌐 {title} ({source.get('source', 'Web')})")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask any legal question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing your legal question..."):
                # Generate answer
                response = st.session_state.legal_rag_system.generate_legal_answer(
                    prompt, 
                    include_web_search=enable_web_search
                )
                
                # Display answer
                answer = response.get('answer', 'No answer generated')
                
                st.markdown(f"""
                <div class="answer-container">
                    {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                sources = response.get('sources', [])
                if sources:
                    with st.expander("📚 Sources Used", expanded=False):
                        for i, source in enumerate(sources, 1):
                            if source.get('type') == 'local':
                                st.write(f"**{i}.** 📄 {source.get('filename', 'Unknown')} (Local Document)")
                            elif source.get('type') == 'web':
                                title = source.get('title', 'Unknown')
                                url = source.get('url', '')
                                if url:
                                    st.write(f"**{i}.** 🌐 [{title}]({url}) ({source.get('source', 'Web')})")
                                else:
                                    st.write(f"**{i}.** 🌐 {title} ({source.get('source', 'Web')})")
                
                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Local Results", response.get('local_results_found', 0))
                with col2:
                    st.metric("Web Results", response.get('web_results_found', 0))
                with col3:
                    st.metric("Total Context", response.get('context_used', 0))
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
    
    # Quick action buttons
    if st.session_state.messages:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("📊 View Analytics", type="secondary"):
                st.info("📊 Analytics dashboard - Coming in next update!")
        
        with col3:
            if st.button("📥 Export Chat", type="secondary"):
                chat_export = []
                for msg in st.session_state.messages:
                    chat_export.append(f"{msg['role'].upper()}: {msg['content']}")
                
                export_text = "\n\n".join(chat_export)
                st.download_button(
                    "Download Chat History",
                    data=export_text,
                    file_name=f"legal_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Sample questions
    if not st.session_state.messages:
        st.markdown("### 💡 Try These Sample Legal Questions:")
        
        sample_questions = [
            "What are my rights if my employer terminates me without notice?",
            "How do I respond to a legal notice for breach of contract?",
            "What are the steps to file a patent application in India?",
            "Can I sue for defamation if someone posts false information about me online?",
            "What are the tax implications of selling inherited property?",
            "How do I incorporate a private limited company in India?",
            "What constitutes criminal negligence in medical practice?",
            "What are the legal requirements for a valid will in India?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = cols[i % 2]
            with col:
                if st.button(f"❓ {question}", key=f"sample_{i}"):
                    # Auto-fill the question
                    st.session_state.auto_question = question
                    st.rerun()
        
        # Handle auto-filled questions
        if hasattr(st.session_state, 'auto_question'):
            st.info(f"🔄 Processing: {st.session_state.auto_question}")
            # Process the auto question
            st.session_state.messages.append({"role": "user", "content": st.session_state.auto_question})
            
            # Generate response
            with st.spinner("🤖 Analyzing your legal question..."):
                response = st.session_state.legal_rag_system.generate_legal_answer(
                    st.session_state.auto_question,
                    include_web_search=enable_web_search
                )
                
                answer = response.get('answer', 'No answer generated')
                sources = response.get('sources', [])
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            
            # Clean up
            del st.session_state.auto_question
            st.rerun()
    
    # Footer with legal disclaimer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <h4>⚖️ Legal Disclaimer</h4>
        <p><strong>This AI system provides general legal information and analysis, not legal advice.</strong></p>
        <p>The information provided should not be relied upon as a substitute for consultation with a qualified attorney. 
        Legal requirements vary by jurisdiction and specific circumstances. Always consult with a licensed legal professional 
        for advice regarding your specific legal situation.</p>
        
        <div style="margin-top: 1rem; padding: 0.5rem; background: white; border-radius: 5px; border: 1px solid #dee2e6;">
            <strong>🚀 Best Legal RAG System v2.0</strong><br>
            FAISS-Based • Error-Free • Production Ready<br>
            <em>Powered by Advanced AI • Real-time Research • Professional Analysis</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


