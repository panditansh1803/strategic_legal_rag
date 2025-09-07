import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import re
import time

# Core dependencies
import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from openai import OpenAI
import google.generativeai as genai

# Document processing
import PyPDF2
import docx
from pathlib import Path

# Web scraping
from bs4 import BeautifulSoup
import requests
import urllib.parse

# NLP processing
try:
    import spacy
    import nltk
except ImportError:
    print("Warning: spacy or nltk not installed. Some NLP features may be limited.")

# UI and visualization
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

@dataclass
class SimilarCase:
    """Represents a similar legal case with strategic analysis"""
    case_id: str
    case_name: str
    facts: str
    legal_issues: List[str]
    court_decision: str
    winning_arguments: List[str]
    losing_arguments: List[str]
    key_evidence: List[str]
    case_outcome: str
    similarity_score: float
    jurisdiction: str
    precedential_value: str
    strategic_lessons: List[str]
    distinguishing_factors: List[str]
    citation: Optional[str] = None
    court: Optional[str] = None
    date_decided: Optional[str] = None
    url: Optional[str] = None

@dataclass
class LegalStrategy:
    """Comprehensive legal strategy for winning a case"""
    strategy_id: str
    case_summary: str
    primary_arguments: List[Dict[str, Any]]
    supporting_precedents: List[SimilarCase]
    evidence_requirements: List[str]
    potential_defenses: List[str]
    counter_arguments: List[Dict[str, Any]]
    procedural_strategy: Dict[str, Any]
    settlement_considerations: Dict[str, Any]
    success_probability: float
    alternative_approaches: List[Dict[str, Any]]
    tactical_recommendations: List[str]
    timeline_strategy: List[Dict[str, Any]]
    cost_benefit_analysis: Dict[str, Any]

class DocumentProcessor:
    """Process and extract information from legal documents"""
    
    def __init__(self):
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            st.warning("spaCy model not found. Some NLP features may be limited.")
            self.nlp = None
        
        # Legal patterns
        self.citation_patterns = [
            r'\d+\s+U\.S\.\s+\d+',  # US Supreme Court
            r'\d+\s+S\.Ct\.\s+\d+',  # Supreme Court Reporter
            r'\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
            r'AIR\s+\d+\s+SC\s+\d+',  # All India Reporter - SC
            r'\d+\s+SCC\s+\d+',  # Supreme Court Cases
            r'\(\d+\)\s+\d+\s+SCC\s+\d+',  # SCC with year
            r'AIR\s+\d+\s+[A-Z]{2,}\s+\d+',  # High Court AIR
        ]
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num}: {e}")
                        continue
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
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
            return ""
    
    def extract_legal_citations(self, text: str) -> List[str]:
        """Extract legal citations from text"""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        return list(set(citations))  # Remove duplicates
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities using NLP"""
        entities = {
            'parties': [],
            'courts': [],
            'judges': [],
            'locations': [],
            'organizations': []
        }
        
        if self.nlp:
            try:
                doc = self.nlp(text[:100000])  # Limit text size for processing
                
                for ent in doc.ents:
                    entity_text = ent.text.strip()
                    if len(entity_text) < 3:  # Skip very short entities
                        continue
                        
                    if ent.label_ == "PERSON":
                        if any(keyword in entity_text.lower() for keyword in ['j.', 'justice', 'judge']):
                            entities['judges'].append(entity_text)
                        else:
                            entities['parties'].append(entity_text)
                    elif ent.label_ == "ORG":
                        if any(keyword in entity_text.lower() for keyword in ['court', 'tribunal']):
                            entities['courts'].append(entity_text)
                        else:
                            entities['organizations'].append(entity_text)
                    elif ent.label_ in ["GPE", "LOC"]:
                        entities['locations'].append(entity_text)
                
                # Remove duplicates and limit results
                for key in entities:
                    entities[key] = list(set(entities[key]))[:10]
                    
            except Exception as e:
                print(f"NLP processing error: {e}")
        
        return entities

class WebLegalSearcher:
    """Search legal databases and free resources online"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def search_indiankanoon(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search IndianKanoon for legal cases"""
        try:
            # Clean and encode query
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            search_url = f"https://indiankanoon.org/search/?formInput={urllib.parse.quote(clean_query)}"
            
            response = self.session.get(search_url, timeout=15)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find result links
            result_divs = soup.find_all('div', class_='result')
            if not result_divs:
                # Try alternative structure
                result_divs = soup.find_all('div', {'class': re.compile(r'result|search')})
            
            count = 0
            for div in result_divs[:max_results*2]:  # Get more to filter
                if count >= max_results:
                    break
                
                link = div.find('a', href=True)
                if link and '/doc/' in link.get('href', ''):
                    title = link.get_text(strip=True)
                    if len(title) > 10:  # Valid title
                        case_url = f"https://indiankanoon.org{link['href']}"
                        
                        # Get snippet if available
                        snippet_div = div.find('div', class_='snippet') or div
                        snippet = snippet_div.get_text(strip=True)[:500]
                        
                        results.append({
                            'title': title,
                            'url': case_url,
                            'snippet': snippet,
                            'source': 'IndianKanoon'
                        })
                        count += 1
            
            return results
            
        except Exception as e:
            print(f"Error searching IndianKanoon: {e}")
            return []
    
    def search_google_scholar(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search Google Scholar for legal articles and cases"""
        try:
            legal_query = f'"{query}" law case court legal'
            search_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote(legal_query)}&hl=en&as_sdt=0,5"
            
            response = self.session.get(search_url, timeout=15)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Find scholar results
            result_divs = soup.find_all('div', class_='gs_r gs_or gs_scl')[:max_results]
            
            for div in result_divs:
                title_elem = div.find('h3', class_='gs_rt')
                if title_elem:
                    title_link = title_elem.find('a')
                    title = title_elem.get_text(strip=True)
                    url = title_link.get('href') if title_link else None
                    
                    # Get snippet
                    snippet_elem = div.find('div', class_='gs_rs')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and len(title) > 10:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet[:500],
                            'source': 'Google Scholar'
                        })
            
            return results
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            return []

class StrategicLegalAnalyzer:
    """Advanced legal analyzer for case strategy and similar case finding"""
    
    def __init__(self, llm_provider: str = "perplexity"):
        self.llm_provider = llm_provider
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_processor = DocumentProcessor()
        self.web_searcher = WebLegalSearcher()
        
        # Initialize vector database
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.get_collection("strategic_legal_db")
        except:
            self.collection = self.chroma_client.create_collection(
                name="strategic_legal_db",
                metadata={"hnsw:space": "cosine"}
            )
        
        self._init_llm()
        self.document_count = 0
        
        # Enhanced strategy templates
        self.strategy_templates = {
            'similar_cases_analysis': """
            You are an expert legal strategist. Analyze the user's case and the similar cases found to provide strategic insights.

            USER'S CASE:
            Facts: {case_facts}
            Legal Issues: {legal_issues}
            Goals: {client_goals}
            Jurisdiction: {jurisdiction}

            SIMILAR CASES FOUND:
            {similar_cases_data}

            Provide comprehensive analysis with:

            ## TOP SIMILAR CASES (Ranked by Strategic Value)

            For each of the top 5 cases, provide:
            - **Case Name & Court**
            - **Similarity Analysis** (why it matches your case)
            - **Winning Strategy Used** (specific arguments that succeeded)
            - **Key Evidence** (what proof was decisive)  
            - **Strategic Lessons** (actionable insights for your case)
            - **Distinguishing Factors** (how to use or distinguish this case)

            ## PRECEDENT HIERARCHY
            - **Binding Authority** (must follow)
            - **Persuasive Authority** (helpful but not binding)
            - **Recent Trends** (how courts are deciding similar cases)

            ## WINNING PATTERNS IDENTIFIED
            - **Common successful arguments** across similar cases
            - **Critical evidence types** that led to victories
            - **Procedural strategies** that worked
            - **Timing considerations** for maximum impact

            ## PITFALLS TO AVOID  
            - **Failed strategies** from similar cases
            - **Evidence gaps** that led to losses
            - **Procedural mistakes** to avoid

            Focus on actionable strategic advice for winning this specific case.
            """,

            'comprehensive_winning_strategy': """
            Create a complete winning strategy based on case analysis and similar successful cases:

            CASE ANALYSIS:
            Facts: {case_facts}
            Legal Issues: {legal_issues}  
            Client Goals: {client_goals}
            Opposing Position: {opposing_arguments}
            Jurisdiction: {jurisdiction}

            SUCCESSFUL SIMILAR CASES:
            {successful_cases_data}

            Create comprehensive winning strategy:

            ## PRIMARY WINNING ARGUMENTS (Priority Order)

            ### 1. STRONGEST ARGUMENT
            - **Legal Theory**: [Primary legal basis]
            - **Supporting Precedents**: [Key cases that support this]
            - **Evidence Required**: [What proof is needed]
            - **Success Probability**: [X%] 
            - **Counter-Argument Response**: [How to handle opponent's challenges]

            ### 2. SECONDARY ARGUMENTS  
            - **Backup Legal Theories**: [Alternative approaches]
            - **Supporting Case Law**: [Additional precedents]
            - **Strategic Value**: [When to use each]

            ## EVIDENCE STRATEGY

            ### Must-Have Evidence (Essential)
            - [Evidence item 1] - Why it's critical
            - [Evidence item 2] - How it supports your case

            ### Supporting Evidence (Strengthening)  
            - [Evidence item 3] - Additional support
            - [Evidence item 4] - Credibility enhancers

            ### Evidence to Challenge (Opponent's Weaknesses)
            - [Opponent evidence 1] - How to attack
            - [Opponent evidence 2] - Credibility challenges

            ## DEFENSIVE STRATEGY

            ### Anticipated Counter-Arguments
            1. **"[Opponent Argument 1]"**
               - Response: [Your counter-response]
               - Supporting Law: [Precedents that help]

            2. **"[Opponent Argument 2]"**  
               - Response: [Your counter-response]
               - Evidence Needed: [What to gather]

            ### Weakness Management
            - **Your Case Weaknesses**: [Honest assessment]
            - **Mitigation Strategies**: [How to minimize impact]
            - **Damage Control**: [If things go wrong]

            ## TACTICAL EXECUTION

            ### Court Strategy
            - **Venue Selection**: [Best court/judge if options]
            - **Timing Strategy**: [When to file, argue, settle]
            - **Presentation Order**: [How to structure arguments]

            ### Settlement Leverage  
            - **Negotiation Strengths**: [What gives you power]
            - **Settlement Value Range**: [Realistic expectations]
            - **Leverage Points**: [When you have maximum power]

            ## SUCCESS ANALYSIS

            ### Probability Assessment
            - **Overall Success Chance**: [X%]
            - **Best Case Outcome**: [Ideal result]
            - **Most Likely Outcome**: [Realistic expectation]
            - **Worst Case Scenario**: [Risk assessment]

            ### Risk-Benefit Analysis
            - **Potential Gains**: [What you can win]
            - **Potential Costs**: [What you might lose]
            - **Strategic Recommendation**: [Proceed/Settle/Modify]

            ## ALTERNATIVE STRATEGIES

            If primary strategy encounters problems:
            1. **Plan B**: [Alternative legal theory]
            2. **Plan C**: [Settlement approach]  
            3. **Exit Strategy**: [How to minimize losses]

            Focus on practical, executable strategies with specific action items.
            """
        }
    
    def _init_llm(self):
        """Initialize the selected LLM"""
        try:
            if self.llm_provider == "perplexity":
                api_key = os.getenv('PERPLEXITY_API_KEY')
                if not api_key:
                    raise ValueError("PERPLEXITY_API_KEY not found in environment variables")
                self.perplexity_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
                )
                
            elif self.llm_provider == "gemini":
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")
                genai.configure(api_key=api_key)
                self.llm_model = genai.GenerativeModel('gemini-pro')
                
            elif self.llm_provider == "openai":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                self.openai_client = OpenAI(api_key=api_key)
                
        except Exception as e:
            st.error(f"Error initializing LLM: {e}")
            raise
    
    def query_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Query the selected LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                if self.llm_provider == "perplexity":
                    response = self.perplexity_client.chat.completions.create(
                        model="llama-3.1-sonar-large-128k-online",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=4000
                    )
                    return response.choices[0].message.content
                    
                elif self.llm_provider == "gemini":
                    response = self.llm_model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,
                            max_output_tokens=4000
                        )
                    )
                    return response.text
                    
                elif self.llm_provider == "openai":
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=4000
                    )
                    return response.choices[0].message.content
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return f"Error querying LLM after {max_retries} attempts: {str(e)}"
    
    def add_document(self, file_path: str) -> str:
        """Add a legal document to the knowledge base"""
        try:
            file_extension = Path(file_path).suffix.lower()
            
            # Extract text based on file type
            if file_extension == '.pdf':
                content = self.document_processor.extract_text_from_pdf(file_path)
            elif file_extension == '.docx':
                content = self.document_processor.extract_text_from_docx(file_path)
            elif file_extension == '.txt':
                content = self.document_processor.extract_text_from_txt(file_path)
            else:
                return f"Unsupported file type: {file_extension}"
            
            if not content.strip():
                return "No text content found in document"
            
            # Extract legal information
            citations = self.document_processor.extract_legal_citations(content)
            entities = self.document_processor.extract_legal_entities(content)
            
            # Generate embeddings
            embedding = self.embedder.encode(content).tolist()
            
            # Generate unique ID
            doc_id = hashlib.md5((content + str(datetime.now())).encode()).hexdigest()[:12]
            file_name = Path(file_path).stem
            
            # Add to vector database
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    'id': doc_id,
                    'title': file_name,
                    'file_path': file_path,
                    'source_type': 'uploaded',
                    'jurisdiction': self._detect_jurisdiction(content),
                    'document_type': self._classify_document_type(content),
                    'citations': json.dumps(citations),
                    'parties': json.dumps(entities.get('parties', [])),
                    'courts': json.dumps(entities.get('courts', [])),
                    'upload_date': datetime.now().isoformat()
                }],
                ids=[doc_id]
            )
            
            self.document_count += 1
            
            return f"Successfully added: {file_name} ({len(content)} chars, {len(citations)} citations)"
            
        except Exception as e:
            return f"Error processing document: {str(e)}"
    
    def find_similar_cases(self, case_facts: str, legal_issues: str, 
                          jurisdiction: str = "India", max_results: int = 10) -> List[SimilarCase]:
        """Find similar cases using hybrid search approach"""
        
        similar_cases = []
        
        # 1. Search local uploaded documents
        local_cases = self._search_local_documents(case_facts, legal_issues, max_results // 2)
        similar_cases.extend(local_cases)
        
        # 2. Search online legal databases  
        online_cases = self._search_online_databases(case_facts, legal_issues, jurisdiction, max_results // 2)
        similar_cases.extend(online_cases)
        
        # 3. Use LLM for additional case discovery
        if self.llm_provider == "perplexity":
            llm_cases = self._llm_enhanced_search(case_facts, legal_issues, jurisdiction)
            similar_cases.extend(llm_cases)
        
        # Remove duplicates and sort by similarity
        unique_cases = {}
        for case in similar_cases:
            case_key = case.case_name.lower().strip()
            if case_key not in unique_cases or case.similarity_score > unique_cases[case_key].similarity_score:
                unique_cases[case_key] = case
        
        # Sort by similarity score and return top results
        sorted_cases = sorted(unique_cases.values(), key=lambda x: x.similarity_score, reverse=True)
        return sorted_cases[:max_results]
    
    def _search_local_documents(self, case_facts: str, legal_issues: str, max_results: int) -> List[SimilarCase]:
        """Search locally uploaded documents"""
        try:
            if self.document_count == 0:
                return []
            
            query_text = f"{case_facts} {legal_issues}"
            query_embedding = self.embedder.encode(query_text).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results, self.document_count),
                include=['documents', 'metadatas', 'distances']
            )
            
            similar_cases = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                similarity_score = max(0, 1 - distance)
                
                similar_cases.append(SimilarCase(
                    case_id=metadata['id'],
                    case_name=metadata['title'],
                    facts=doc[:300] + "...",
                    legal_issues=[legal_issues],
                    court_decision='Unknown',
                    winning_arguments=[],
                    losing_arguments=[],
                    key_evidence=[],
                    case_outcome='Unknown',
                    similarity_score=similarity_score,
                    jurisdiction=metadata.get('jurisdiction', 'Unknown'),
                    precedential_value='Unknown',
                    strategic_lessons=[],
                    distinguishing_factors=[]
                ))
            
            return similar_cases
            
        except Exception as e:
            print(f"Error searching local documents: {e}")
            return []
    
    def _search_online_databases(self, case_facts: str, legal_issues: str, 
                                jurisdiction: str, max_results: int) -> List[SimilarCase]:
        """Search online legal databases"""
        online_cases = []
        
        try:
            # Search IndianKanoon for Indian cases
            if jurisdiction.lower() in ['india', 'indian']:
                ik_results = self.web_searcher.search_indiankanoon(f"{legal_issues} {case_facts}", max_results)
                for result in ik_results:
                    online_cases.append(SimilarCase(
                        case_id=f"ik_{uuid.uuid4().hex[:8]}",
                        case_name=result['title'],
                        facts=result['snippet'],
                        legal_issues=[legal_issues],
                        court_decision='Unknown',
                        winning_arguments=[],
                        losing_arguments=[],
                        key_evidence=[],
                        case_outcome='Unknown',
                        similarity_score=self._calculate_similarity_score(case_facts, legal_issues, result['snippet']),
                        jurisdiction='India',
                        precedential_value='Persuasive',
                        strategic_lessons=[],
                        distinguishing_factors=[],
                        url=result['url']
                    ))
            
        except Exception as e:
            print(f"Error searching online databases: {e}")
        
        return online_cases
    
    def _llm_enhanced_search(self, case_facts: str, legal_issues: str, jurisdiction: str) -> List[SimilarCase]:
        """Use LLM (Perplexity) to find additional similar cases"""
        try:
            search_prompt = f"""
            Find 3-5 legal cases similar to this situation:

            Case Facts: {case_facts}
            Legal Issues: {legal_issues}  
            Jurisdiction Focus: {jurisdiction}

            For each case, provide:
            - Case name and citation
            - Key facts (brief)
            - Court decision/outcome
            - Winning arguments used
            - Key evidence that mattered
            - Strategic lessons for similar cases

            Focus on cases that would provide strategic insights for winning a similar case.
            """

            llm_response = self.query_llm(search_prompt)
            return self._parse_llm_cases(llm_response, legal_issues, jurisdiction)
            
        except Exception as e:
            print(f"Error in LLM enhanced search: {e}")
            return []
    
    def _parse_llm_cases(self, llm_response: str, legal_issues: str, jurisdiction: str) -> List[SimilarCase]:
        """Parse LLM response to extract case information"""
        llm_cases = []
        
        # Split response into case sections
        case_sections = re.split(r'\n\s*\d+\.\s*', llm_response)
        
        for i, section in enumerate(case_sections[1:], 1):
            if len(section.strip()) > 50:
                lines = section.strip().split('\n')
                case_name = lines[0].strip()
                case_content = '\n'.join(lines[1:])
                
                llm_cases.append(SimilarCase(
                    case_id=f"llm_{uuid.uuid4().hex[:8]}",
                    case_name=case_name[:100],
                    facts=case_content[:300],
                    legal_issues=[legal_issues],
                    court_decision='Unknown',
                    winning_arguments=[],
                    losing_arguments=[],
                    key_evidence=[],
                    case_outcome='Unknown',
                    similarity_score=0.7,
                    jurisdiction=jurisdiction,
                    precedential_value='Informational',
                    strategic_lessons=[],
                    distinguishing_factors=[]
                ))
        
        return llm_cases[:5]
    
    def _calculate_similarity_score(self, case_facts: str, legal_issues: str, comparison_text: str) -> float:
        """Calculate similarity score between user case and found case"""
        try:
            user_text = f"{case_facts} {legal_issues}"
            user_embedding = self.embedder.encode(user_text)
            comparison_embedding = self.embedder.encode(comparison_text)
            
            # Simple cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(user_embedding, comparison_embedding))
            magnitude_a = sum(a * a for a in user_embedding) ** 0.5
            magnitude_b = sum(b * b for b in comparison_embedding) ** 0.5
            
            if magnitude_a == 0 or magnitude_b == 0:
                return 0.0
            
            similarity = dot_product / (magnitude_a * magnitude_b)
            return max(0, min(1, similarity))
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.5  # Default similarity
    
    def create_winning_strategy(self, case_facts: str, legal_issues: str, client_goals: str,
                              jurisdiction: str = "India", opposing_arguments: str = "") -> Tuple[Dict[str, Any], str]:
        """Create comprehensive winning strategy for the case"""
        
        similar_cases = self.find_similar_cases(case_facts, legal_issues, jurisdiction, max_results=8)
        
        cases_data = []
        for case in similar_cases[:5]:
            case_summary = f"""
**{case.case_name}** (Similarity: {case.similarity_score:.1%})
- Jurisdiction: {case.jurisdiction}
- Facts: {case.facts[:200]}...
- Outcome: {case.case_outcome}
"""
            cases_data.append(case_summary)
        
        # Create comprehensive strategy prompt
        strategy_prompt = self.strategy_templates['comprehensive_winning_strategy'].format(
            case_facts=case_facts,
            legal_issues=legal_issues,
            client_goals=client_goals,
            opposing_arguments=opposing_arguments or "Not specified",
            jurisdiction=jurisdiction,
            successful_cases_data='\n'.join(cases_data)
        )
        
        # Get strategy analysis from LLM
        strategy_analysis = self.query_llm(strategy_prompt)
        
        # Calculate success probability based on similar cases
        success_scores = [case.similarity_score for case in similar_cases if case.case_outcome.lower() in ['won', 'victory', 'favorable']]
        avg_success_probability = sum(success_scores) / len(success_scores) if success_scores else 0.5
        
        # Create structured strategy object
        strategy_data = {
            'strategy_id': f"strategy_{uuid.uuid4().hex[:8]}",
            'case_summary': f"Facts: {case_facts[:200]}... Goals: {client_goals}",
            'similar_cases_count': len(similar_cases),
            'top_similar_cases': similar_cases[:3],
            'success_probability': avg_success_probability,
            'jurisdiction': jurisdiction,
            'created_date': datetime.now().isoformat(),
            'strategy_analysis': strategy_analysis
        }
        
        return strategy_data, strategy_analysis
    
    def calculate_success_probability(self, strength_factors: Dict[str, int], 
                                    challenge_factors: Dict[str, int]) -> Dict[str, Any]:
        """Calculate case success probability based on various factors"""
        
        # Weighted calculation of strength factors
        strength_weights = {
            'precedent_strength': 0.25,
            'evidence_quality': 0.25,
            'legal_merit': 0.20,
            'witness_credibility': 0.15,
            'attorney_quality': 0.10,
            'case_preparation': 0.05
        }
        
        challenge_weights = {
            'opponent_resources': 0.20,
            'case_complexity': 0.20,
            'procedural_hurdles': 0.20,
            'judge_bias_risk': 0.15,
            'public_opinion': 0.10,
            'timeline_pressure': 0.10,
            'cost_constraints': 0.05
        }
        
        # Calculate weighted scores
        strength_score = sum(
            strength_factors.get(factor, 5) * weight 
            for factor, weight in strength_weights.items()
        )
        
        challenge_score = sum(
            challenge_factors.get(factor, 5) * weight
            for factor, weight in challenge_weights.items()
        )
        
        # Base probability calculation
        base_probability = (strength_score / 10) * 100
        
        # Adjust for challenges
        challenge_adjustment = (challenge_score - 5) * 2
        adjusted_probability = base_probability - challenge_adjustment
        
        # Ensure probability stays within reasonable bounds
        final_probability = max(5, min(95, adjusted_probability))
        
        # Determine confidence level
        if final_probability >= 75:
            confidence = "High"
            recommendation = "Pursue aggressively"
        elif final_probability >= 55:
            confidence = "Moderate" 
            recommendation = "Proceed with caution"
        else:
            confidence = "Low"
            recommendation = "Consider settlement"
        
        return {
            'success_probability': final_probability,
            'confidence_level': confidence,
            'recommendation': recommendation,
            'strength_score': strength_score,
            'challenge_score': challenge_score,
            'settlement_value': int(final_probability * 0.75),
            'analysis': {
                'strengths': [f for f, v in strength_factors.items() if v >= 7],
                'weaknesses': [f for f, v in strength_factors.items() if v <= 4],
                'major_challenges': [f for f, v in challenge_factors.items() if v >= 7],
                'advantages': [f for f, v in challenge_factors.items() if v <= 4]
            }
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the legal database"""
        try:
            all_docs = self.collection.get(include=['metadatas'])
            
            if not all_docs['metadatas']:
                return {
                    'total_documents': 0,
                    'document_types': {},
                    'jurisdictions': {},
                    'upload_dates': []
                }
            
            # Analyze metadata
            doc_types = {}
            jurisdictions = {}
            upload_dates = []
            
            for metadata in all_docs['metadatas']:
                # Document types
                doc_type = metadata.get('document_type', 'Unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Jurisdictions
                jurisdiction = metadata.get('jurisdiction', 'Unknown')
                jurisdictions[jurisdiction] = jurisdictions.get(jurisdiction, 0) + 1
                
                # Upload dates
                upload_date = metadata.get('upload_date')
                if upload_date:
                    upload_dates.append(upload_date)
            
            return {
                'total_documents': len(all_docs['metadatas']),
                'document_types': doc_types,
                'jurisdictions': jurisdictions,
                'upload_dates': upload_dates,
                'last_updated': max(upload_dates) if upload_dates else None
            }
            
        except Exception as e:
            return {'error': str(e), 'total_documents': 0}
    
    def _detect_jurisdiction(self, content: str) -> str:
        """Detect jurisdiction from document content"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['supreme court of india', 'high court', 'indian', 'constitution of india']):
            return 'India'
        elif any(term in content_lower for term in ['u.s. supreme court', 'federal court', 'united states']):
            return 'United States'
        elif any(term in content_lower for term in ['house of lords', 'court of appeal', 'england', 'wales']):
            return 'United Kingdom'
        elif 'singapore' in content_lower:
            return 'Singapore'
        else:
            return 'Unknown'
    
    def _classify_document_type(self, content: str) -> str:
        """Classify document type based on content"""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['judgment', 'held', 'coram', 'bench', 'decided']):
            return 'case'
        elif any(term in content_lower for term in ['section', 'chapter', 'act', 'statute', 'code']):
            return 'statute'
        elif any(term in content_lower for term in ['brief', 'petition', 'pleading', 'motion']):
            return 'brief'
        elif any(term in content_lower for term in ['law review', 'article', 'commentary', 'analysis']):
            return 'article'
        else:
            return 'document'

# Complete Streamlit Application
def create_complete_strategic_legal_app():
    """Create the complete strategic legal application"""
    
    st.set_page_config(
        page_title="Strategic Legal RAG - Complete System",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .success-high { background-color: #d4edda; border-color: #28a745; }
    .success-medium { background-color: #fff3cd; border-color: #ffc107; }
    .success-low { background-color: #f8d7da; border-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">⚖️ Strategic Legal RAG Agent</h1>
        <p style="color: #e8f4fd; margin: 0;">Find Similar Cases • Develop Winning Strategies • Maximize Success Probability</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'strategic_agent' not in st.session_state:
        with st.spinner("Initializing Strategic Legal RAG Agent..."):
            try:
                # LLM provider selection
                llm_provider = st.sidebar.selectbox(
                    "Select LLM Provider",
                    ["perplexity", "gemini", "openai"],
                    help="Perplexity recommended for real-time legal research"
                )
                
                st.session_state.strategic_agent = StrategicLegalAnalyzer(llm_provider)
                st.success(f"Agent initialized with {llm_provider.title()} LLM")
                
            except Exception as e:
                st.error(f"Error initializing agent: {e}")
                st.stop()
    
    # Sidebar with configuration and document management
    with st.sidebar:
        st.header("Configuration")
        
        # API Key status
        if st.session_state.strategic_agent.llm_provider == "perplexity":
            api_status = "✅" if os.getenv('PERPLEXITY_API_KEY') else "❌"
            st.write(f"Perplexity API: {api_status}")
        elif st.session_state.strategic_agent.llm_provider == "gemini":
            api_status = "✅" if os.getenv('GOOGLE_API_KEY') else "❌"
            st.write(f"Gemini API: {api_status}")
        else:
            api_status = "✅" if os.getenv('OPENAI_API_KEY') else "❌"
            st.write(f"OpenAI API: {api_status}")
        
        st.divider()
        
        # Document upload section
        st.header("Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Legal Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload cases, statutes, articles, or legal briefs"
        )
        
        if uploaded_files:
            upload_progress = st.progress(0)
            status_container = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_container.text(f"Processing: {uploaded_file.name}")
                
                # Save temporary file
                temp_path = f"temp_{uploaded_file.name}"
                try:
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Add to database
                    result = st.session_state.strategic_agent.add_document(temp_path)
                    
                    if "Successfully" in result:
                        st.success(f"{uploaded_file.name}: Added successfully")
                    else:
                        st.error(f"{uploaded_file.name}: {result}")
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                # Update progress
                progress = (i + 1) / len(uploaded_files)
                upload_progress.progress(progress)
            
            status_container.text("Upload complete!")
            time.sleep(1)
            st.rerun()
        
        st.divider()
        
        # Database statistics
        st.header("Database Stats")
        stats = st.session_state.strategic_agent.get_database_stats()
        
        if stats.get('total_documents', 0) > 0:
            st.metric("Total Documents", stats['total_documents'])
            
            # Document type breakdown
            if stats.get('document_types'):
                st.write("**Document Types:**")
                for doc_type, count in stats['document_types'].items():
                    st.write(f"• {doc_type.title()}: {count}")
            
            # Jurisdiction breakdown
            if stats.get('jurisdictions'):
                st.write("**Jurisdictions:**")
                for jurisdiction, count in stats['jurisdictions'].items():
                    st.write(f"• {jurisdiction}: {count}")
        else:
            st.info("No documents uploaded yet")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Similar Cases",
        "Winning Strategy", 
        "Success Calculator",
        "Analytics Dashboard"
    ])
    
    # Tab 1: Similar Cases Finder
    with tab1:
        st.header("Find Cases Similar to Yours")
        st.markdown("*Discover cases with matching facts and winning strategies*")
        
        # Input form
        col1, col2 = st.columns([3, 2])
        
        with col1:
            case_facts = st.text_area(
                "**Your Case Facts**",
                placeholder="Describe what happened:\n• Key events and timeline\n• Parties involved\n• Damages or disputes\n• Relevant circumstances",
                height=150,
                help="Be specific about facts that might be legally relevant"
            )
            
            legal_issues = st.text_area(
                "**Legal Issues**",
                placeholder="What legal questions need resolution:\n• Contract breach?\n• Negligence claims?\n• Constitutional violations?\n• Statutory interpretations?",
                height=120,
                help="Focus on the core legal disputes"
            )
            
            client_goals = st.text_area(
                "**Your Objectives**",
                placeholder="What you want to achieve:\n• Monetary compensation?\n• Injunctive relief?\n• Case dismissal?\n• Specific performance?",
                height=100,
                help="Clear objectives help focus the search"
            )
        
        with col2:
            st.subheader("Search Parameters")
            
            jurisdiction = st.selectbox(
                "**Primary Jurisdiction**",
                ["India", "United States", "United Kingdom", "Singapore", "International"],
                help="Focus on specific legal system"
            )
            
            max_cases = st.slider(
                "**Number of Cases to Find**",
                min_value=5,
                max_value=25,
                value=10,
                help="More cases = broader perspective, fewer = focused results"
            )
        
        # Search button and results
        if st.button("Find Similar Cases", type="primary", use_container_width=True):
            if case_facts and legal_issues:
                with st.spinner("Searching legal databases for similar cases..."):
                    similar_cases = st.session_state.strategic_agent.find_similar_cases(
                        case_facts, legal_issues, jurisdiction, max_cases
                    )
                
                if similar_cases:
                    st.subheader(f"Found {len(similar_cases)} Similar Cases")
                    
                    # Display cases
                    for i, case in enumerate(similar_cases, 1):
                        # Determine similarity color coding
                        if case.similarity_score >= 0.8:
                            similarity_color = "🟢"
                            similarity_text = "Very High"
                        elif case.similarity_score >= 0.6:
                            similarity_color = "🟡"
                            similarity_text = "High"
                        elif case.similarity_score >= 0.4:
                            similarity_color = "🟠"
                            similarity_text = "Moderate"
                        else:
                            similarity_color = "🔴"
                            similarity_text = "Low"
                        
                        with st.expander(
                            f"{similarity_color} **#{i} - {case.case_name}** "
                            f"(Similarity: {case.similarity_score:.1%} - {similarity_text})"
                        ):
                            # Two-column layout for case details
                            col1, col2 = st.columns([3, 2])
                            
                            with col1:
                                st.markdown("**Case Facts:**")
                                st.write(case.facts)
                                
                                if case.legal_issues:
                                    st.markdown("**Legal Issues:**")
                                    for issue in case.legal_issues:
                                        st.write(f"• {issue}")
                                
                                st.markdown("**Court & Jurisdiction:**")
                                court_info = f"{case.court or 'Unknown Court'}, {case.jurisdiction}"
                                if case.precedential_value != 'Unknown':
                                    court_info += f" ({case.precedential_value} Authority)"
                                st.write(court_info)
                                
                                if case.citation:
                                    st.markdown(f"**Citation:** {case.citation}")
                                
                                if case.date_decided:
                                    st.markdown(f"**Date:** {case.date_decided}")
                            
                            with col2:
                                # Winning elements
                                if case.winning_arguments:
                                    st.markdown("**Winning Arguments:**")
                                    for arg in case.winning_arguments:
                                        st.success(f"✓ {arg}")
                                
                                if case.key_evidence:
                                    st.markdown("**Key Evidence:**")
                                    for evidence in case.key_evidence:
                                        st.info(f"📝 {evidence}")
                                
                                if case.case_outcome != 'Unknown':
                                    outcome_color = "success" if case.case_outcome.lower() in ['won', 'victory', 'favorable'] else "warning"
                                    st.markdown("**Outcome:**")
                                    if outcome_color == "success":
                                        st.success(f"🏆 {case.case_outcome}")
                                    else:
                                        st.warning(f"⚖️ {case.case_outcome}")
                                
                                # External link if available
                                if case.url:
                                    st.markdown(f"[📖 **Read Full Case**]({case.url})")
                else:
                    st.warning("No similar cases found. Try different search terms or add more documents to your database.")
            else:
                st.warning("Please provide both case facts and legal issues to search for similar cases.")
    
    # Tab 2: Winning Strategy
    with tab2:
        st.header("Develop Your Winning Strategy")
        st.markdown("*Create comprehensive legal strategy based on similar successful cases*")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            strategy_case_facts = st.text_area(
                "**Case Facts**",
                value=st.session_state.get('last_case_facts', ''),
                placeholder="Describe your case facts in detail...",
                height=150
            )
            
            strategy_legal_issues = st.text_area(
                "**Legal Issues**",
                value=st.session_state.get('last_legal_issues', ''),
                placeholder="What are the main legal questions?",
                height=120
            )
            
            strategy_client_goals = st.text_area(
                "**Client Goals**",
                value=st.session_state.get('last_client_goals', ''),
                placeholder="What do you want to achieve?",
                height=100
            )
            
            opposing_arguments = st.text_area(
                "**Opposing Arguments (Optional)**",
                placeholder="What arguments will the other side make?",
                height=80
            )
        
        with col2:
            st.subheader("Strategy Parameters")
            
            strategy_jurisdiction = st.selectbox(
                "**Jurisdiction**",
                ["India", "United States", "United Kingdom", "Singapore", "International"],
                key="strategy_jurisdiction"
            )
            
            case_type = st.selectbox(
                "**Case Type**",
                ["Civil Litigation", "Criminal Defense", "Corporate Law", "Contract Dispute", 
                 "Tort Claims", "Constitutional Law", "IP Law", "Employment Law"]
            )
            
            urgency = st.selectbox(
                "**Case Urgency**",
                ["High - Court date approaching", "Medium - Active case", "Low - Early planning"]
            )
        
        if st.button("Generate Winning Strategy", type="primary", use_container_width=True):
            if strategy_case_facts and strategy_legal_issues and strategy_client_goals:
                # Store for reuse
                st.session_state['last_case_facts'] = strategy_case_facts
                st.session_state['last_legal_issues'] = strategy_legal_issues
                st.session_state['last_client_goals'] = strategy_client_goals
                
                with st.spinner("Analyzing similar cases and developing winning strategy..."):
                    strategy_data, strategy_analysis = st.session_state.strategic_agent.create_winning_strategy(
                        strategy_case_facts, 
                        strategy_legal_issues, 
                        strategy_client_goals,
                        strategy_jurisdiction,
                        opposing_arguments
                    )
                
                # Display strategy
                st.subheader("Your Complete Winning Strategy")
                
                # Success probability indicator
                success_prob = strategy_data['success_probability']
                if success_prob >= 0.75:
                    prob_color = "success"
                    prob_icon = "🟢"
                elif success_prob >= 0.55:
                    prob_color = "warning"
                    prob_icon = "🟡"
                else:
                    prob_color = "error"
                    prob_icon = "🔴"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Success Probability", f"{success_prob:.1%}", delta=None)
                with col2:
                    st.metric("Similar Cases Found", strategy_data['similar_cases_count'])
                with col3:
                    st.metric("Jurisdiction", strategy_data['jurisdiction'])
                
                # Strategy analysis
                st.markdown("### Strategic Analysis")
                st.markdown(strategy_analysis)
                
                # Top similar cases reference
                if strategy_data['top_similar_cases']:
                    st.markdown("### Key Supporting Cases")
                    for i, case in enumerate(strategy_data['top_similar_cases'], 1):
                        st.markdown(f"**{i}. {case.case_name}** (Similarity: {case.similarity_score:.1%})")
                        st.write(f"Facts: {case.facts[:150]}...")
                        if case.url:
                            st.markdown(f"[Read Full Case]({case.url})")
                        st.markdown("---")
            else:
                st.warning("Please provide case facts, legal issues, and client goals to generate a strategy.")
    
    # Tab 3: Success Calculator
    with tab3:
        st.header("Case Success Probability Calculator")
        st.markdown("*Evaluate your chances of winning based on multiple factors*")
        
        st.markdown("### Rate Each Factor (1 = Very Weak, 10 = Very Strong)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strength Factors")
            
            strength_factors = {}
            strength_factors['precedent_strength'] = st.slider(
                "**Legal Precedent Strength**",
                min_value=1, max_value=10, value=5,
                help="How strong is the legal precedent supporting your case?"
            )
            
            strength_factors['evidence_quality'] = st.slider(
                "**Evidence Quality**",
                min_value=1, max_value=10, value=5,
                help="How strong and convincing is your evidence?"
            )
            
            strength_factors['legal_merit'] = st.slider(
                "**Legal Merit**",
                min_value=1, max_value=10, value=5,
                help="How strong are your legal arguments?"
            )
            
            strength_factors['witness_credibility'] = st.slider(
                "**Witness Credibility**",
                min_value=1, max_value=10, value=5,
                help="How credible and reliable are your witnesses?"
            )
            
            strength_factors['attorney_quality'] = st.slider(
                "**Attorney Quality**",
                min_value=1, max_value=10, value=7,
                help="Quality of legal representation"
            )
            
            strength_factors['case_preparation'] = st.slider(
                "**Case Preparation**",
                min_value=1, max_value=10, value=6,
                help="How well prepared is your case?"
            )
        
        with col2:
            st.subheader("Challenge Factors")
            
            challenge_factors = {}
            challenge_factors['opponent_resources'] = st.slider(
                "**Opponent Resources**",
                min_value=1, max_value=10, value=5,
                help="How well-funded/resourced is the opposing side?"
            )
            
            challenge_factors['case_complexity'] = st.slider(
                "**Case Complexity**",
                min_value=1, max_value=10, value=5,
                help="How complex is your case legally/factually?"
            )
            
            challenge_factors['procedural_hurdles'] = st.slider(
                "**Procedural Hurdles**",
                min_value=1, max_value=10, value=5,
                help="How many procedural challenges do you face?"
            )
            
            challenge_factors['judge_bias_risk'] = st.slider(
                "**Judge Bias Risk**",
                min_value=1, max_value=10, value=5,
                help="Risk of unfavorable judge/jury bias"
            )
            
            challenge_factors['public_opinion'] = st.slider(
                "**Public Opinion Impact**",
                min_value=1, max_value=10, value=5,
                help="How might public opinion affect the case?"
            )
            
            challenge_factors['timeline_pressure'] = st.slider(
                "**Timeline Pressure**",
                min_value=1, max_value=10, value=5,
                help="How tight are your deadlines/timeline?"
            )
        
        if st.button("Calculate Success Probability", type="primary", use_container_width=True):
            probability_analysis = st.session_state.strategic_agent.calculate_success_probability(
                strength_factors, challenge_factors
            )
            
            # Display results
            st.markdown("### Success Analysis Results")
            
            # Main probability display
            prob = probability_analysis['success_probability']
            confidence = probability_analysis['confidence_level']
            
            if prob >= 75:
                color = "success"
                bgcolor = "success-high"
            elif prob >= 55:
                color = "warning"
                bgcolor = "success-medium"
            else:
                color = "error"
                bgcolor = "success-low"
            
            # Results display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {bgcolor}">
                    <h2 style="margin: 0;">{prob:.0f}%</h2>
                    <p style="margin: 0;">Success Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">{confidence}</h3>
                    <p style="margin: 0;">Confidence Level</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0;">{probability_analysis['settlement_value']}%</h3>
                    <p style="margin: 0;">Settlement Value</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommendation
            st.markdown(f"### Recommendation: {probability_analysis['recommendation']}")
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Your Advantages")
                analysis = probability_analysis['analysis']
                
                if analysis['strengths']:
                    for strength in analysis['strengths']:
                        st.success(f"✓ Strong {strength.replace('_', ' ').title()}")
                
                if analysis['advantages']:
                    for advantage in analysis['advantages']:
                        st.success(f"✓ Low {advantage.replace('_', ' ').title()}")
            
            with col2:
                st.markdown("#### Areas of Concern")
                
                if analysis['weaknesses']:
                    for weakness in analysis['weaknesses']:
                        st.error(f"⚠ Weak {weakness.replace('_', ' ').title()}")
                
                if analysis['major_challenges']:
                    for challenge in analysis['major_challenges']:
                        st.error(f"⚠ High {challenge.replace('_', ' ').title()}")
            
            # Visual probability chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Success Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Analytics Dashboard
    with tab4:
        st.header("Legal Analytics Dashboard")
        st.markdown("*Visualize patterns and insights from your legal database*")
        
        # Database overview
        stats = st.session_state.strategic_agent.get_database_stats()
        
        if stats.get('total_documents', 0) > 0:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", stats['total_documents'])
            with col2:
                total_types = len(stats.get('document_types', {}))
                st.metric("Document Types", total_types)
            with col3:
                total_jurisdictions = len(stats.get('jurisdictions', {}))
                st.metric("Jurisdictions", total_jurisdictions)
            with col4:
                if stats.get('last_updated'):
                    last_update = stats['last_updated'][:10]  # Just the date
                    st.metric("Last Updated", last_update)
                else:
                    st.metric("Last Updated", "Never")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Document types pie chart
                if stats.get('document_types'):
                    fig_types = px.pie(
                        values=list(stats['document_types'].values()),
                        names=list(stats['document_types'].keys()),
                        title="Document Types Distribution"
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
            
            with col2:
                # Jurisdictions bar chart
                if stats.get('jurisdictions'):
                    fig_jurisdictions = px.bar(
                        x=list(stats['jurisdictions'].keys()),
                        y=list(stats['jurisdictions'].values()),
                        title="Documents by Jurisdiction"
                    )
                    fig_jurisdictions.update_layout(
                        xaxis_title="Jurisdiction",
                        yaxis_title="Number of Documents"
                    )
                    st.plotly_chart(fig_jurisdictions, use_container_width=True)
            
            # Upload timeline
            if stats.get('upload_dates'):
                st.subheader("Document Upload Timeline")
                
                # Convert dates and create timeline
                upload_dates = []
                for date_str in stats['upload_dates']:
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        upload_dates.append(date_obj.date())
                    except:
                        continue
                
                if upload_dates:
                    # Count documents by date
                    date_counts = {}
                    for date in upload_dates:
                        date_counts[date] = date_counts.get(date, 0) + 1
                    
                    # Create timeline chart
                    fig_timeline = px.line(
                        x=list(date_counts.keys()),
                        y=list(date_counts.values()),
                        title="Document Upload Activity",
                        markers=True
                    )
                    fig_timeline.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Documents Uploaded"
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Search performance insights
            st.subheader("Search Performance Insights")
            
            if 'search_history' not in st.session_state:
                st.session_state.search_history = []
            
            if st.session_state.search_history:
                # Display recent searches
                st.markdown("**Recent Searches:**")
                for search in st.session_state.search_history[-5:]:
                    st.write(f"• {search[:100]}...")
            else:
                st.info("No searches performed yet. Use the Similar Cases tab to start building search history.")
            
            # Tips for better results
            st.subheader("Tips for Better Results")
            
            tips = [
                "Upload more diverse case types to improve search accuracy",
                "Include landmark cases from your jurisdiction for better precedent matching",
                "Add statutory materials and legal articles for comprehensive analysis",
                "Use specific legal terminology in your search queries",
                "Regularly update your database with recent cases and rulings"
            ]
            
            for tip in tips:
                st.info(f"💡 {tip}")
        
        else:
            st.info("No documents in database yet. Upload some legal documents to see analytics.")
            
            # Show demo data visualization
            st.subheader("Sample Analytics (Demo Data)")
            
            # Demo charts
            demo_data = {
                'Document Types': {'Cases': 45, 'Statutes': 20, 'Articles': 25, 'Briefs': 10},
                'Jurisdictions': {'India': 60, 'United States': 25, 'United Kingdom': 10, 'Singapore': 5}
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_demo_types = px.pie(
                    values=list(demo_data['Document Types'].values()),
                    names=list(demo_data['Document Types'].keys()),
                    title="Document Types (Demo)"
                )
                st.plotly_chart(fig_demo_types, use_container_width=True)
            
            with col2:
                fig_demo_jurisdictions = px.bar(
                    x=list(demo_data['Jurisdictions'].keys()),
                    y=list(demo_data['Jurisdictions'].values()),
                    title="Jurisdictions (Demo)"
                )
                st.plotly_chart(fig_demo_jurisdictions, use_container_width=True)

# Main execution
if __name__ == "__main__":
    try:
        create_complete_strategic_legal_app()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your API keys and dependencies.")

# Installation instructions and requirements
"""
INSTALLATION INSTRUCTIONS:

1. Install required packages:
   pip install streamlit sentence-transformers chromadb openai google-generativeai
   pip install PyPDF2 python-docx beautifulsoup4 requests plotly pandas
   pip install spacy nltk python-dotenv

2. Install spaCy model:
   python -m spacy download en_core_web_sm

3. Set up API keys in .env file:
   PERPLEXITY_API_KEY=your_perplexity_key
   GOOGLE_API_KEY=your_gemini_key  
   OPENAI_API_KEY=your_openai_key

4. Run the application:
   streamlit run strategic_legal_rag.py

FEATURES:
✅ Hybrid search (local + web + LLM)
✅ Similar case finding with similarity scoring
✅ Winning strategy generation
✅ Success probability calculator
✅ Multi-jurisdiction support (India, US, UK, Singapore)
✅ Multiple LLM providers (Perplexity, Gemini, OpenAI)
✅ Document processing (PDF, DOCX, TXT)
✅ Legal citation extraction
✅ Analytics dashboard
✅ Web scraping (IndianKanoon, Google Scholar)
✅ Strategic analysis and recommendations
✅ Interactive visualizations
✅ Case outcome prediction
✅ Evidence strategy planning
✅ Counter-argument preparation

The agent works in 3 modes:
1. Immediate use with web search (no documents needed)
2. Enhanced with uploaded documents
3. Full power with comprehensive legal database

Perfect for law students and legal professionals who need:
- Strategic case analysis
- Similar case research  
- Winning argument identification
- Success probability assessment
- Evidence planning
- Counter-strategy development
"""