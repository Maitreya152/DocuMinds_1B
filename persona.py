import sys
import json
import re
from datetime import datetime
import time
from typing import List, Dict, Any
import numpy as np
from collections import Counter
import math
import fitz
import os
from parsing import run_parser
import spacy
import logging

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

nlp = spacy.load("/app/spacy_models/en_core_web_sm/en_core_web_sm-3.8.0")

try:
    import nltk
    nltk.data.path.append("/app/nltk_data")
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
    STEMMER = PorterStemmer()
    
except ImportError:
    NLTK_AVAILABLE = False
    STOP_WORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
    STEMMER = None
    print("Warning: nltk not available. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

class BM25:
    """BM25 implementation for fast text similarity scoring"""
    
    def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = []
        self.doc_freqs = []
        self.idf = {}
        self.avgdl = 0
        
        self._preprocess()
    
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization using NLTK"""
        if NLTK_AVAILABLE:
            try:
                # Use NLTK tokenization
                tokens = word_tokenize(text.lower())
                # Filter out non-alphabetic tokens and stopwords
                tokens = [token for token in tokens if token.isalpha() and token not in STOP_WORDS]
                # Apply stemming if available
                if STEMMER:
                    tokens = [STEMMER.stem(token) for token in tokens]
                return tokens
            except Exception as e:
                print(f"NLTK tokenization failed: {e}, falling back to regex")
        
        # Fallback to regex-based tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]
    
    def _preprocess(self):
        """Preprocess corpus for BM25 scoring"""
        tokenized_corpus = []
        
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.doc_freqs.append(Counter(tokens))
        
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Calculate IDF
        all_tokens = set()
        for doc_tokens in tokenized_corpus:
            all_tokens.update(doc_tokens)
        
        for token in all_tokens:
            containing_docs = sum(1 for doc_freq in self.doc_freqs if token in doc_freq)
            self.idf[token] = math.log((len(self.corpus) - containing_docs + 0.5) / (containing_docs + 0.5))
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query against a document"""
        query_tokens = self._tokenize(query)
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]
        
        score = 0
        for token in query_tokens:
            if token in doc_freq:
                tf = doc_freq[token]
                idf = self.idf.get(token, 0)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
        
        return score
    
    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for query against all documents"""
        return [self.score(query, i) for i in range(len(self.corpus))]

class DocumentSection:
    """Represents a document section with metadata"""
    
    def __init__(self, document: str, section_title: str, content: str, page_number: int):
        self.document = document
        self.section_title = section_title
        self.content = content
        self.page_number = page_number
        self.importance_rank = None
        self.bm25_score = 0.0
        self.semantic_score = 0.0
        self.persona_score = 0.0
        self.final_score = 0.0

class IntelligentDocumentAnalyst:
    """Main system for analyzing and prioritizing document sections"""
    
    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('/app/hf_models/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf', local_files_only=True)
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.embedding_model = None
    
    def _split_text_into_chunks(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into chunks that fit within token limit"""
        # Simple approximation: ~4 characters per token for English text
        chars_per_token = 5
        max_chars = max_tokens * chars_per_token
        
        # Split by sentences first to maintain coherence
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed limit, start new chunk
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # If no chunks were created (very short text), return the original
        if not chunks:
            chunks = [text[:max_chars]]
        
        return chunks
    
    def _compute_chunk_similarity_to_title(self, chunk: str, title: str) -> float:
        """Compute similarity between a chunk and section title using miniLM embeddings"""
        if not title or not chunk or not self.embedding_model:
            return 0.1  # Default small weight if no model or empty inputs
        
        try:
            # Compute embeddings for both chunk and title
            chunk_embedding = self.embedding_model.encode([chunk])[0]
            title_embedding = self.embedding_model.encode([title])[0]
            
            # Calculate cosine similarity
            chunk_norm = np.linalg.norm(chunk_embedding)
            title_norm = np.linalg.norm(title_embedding)
            
            if chunk_norm > 0 and title_norm > 0:
                similarity = np.dot(chunk_embedding, title_embedding) / (chunk_norm * title_norm)
                # Convert to non-negative and add small offset to avoid zero weights
                return max(0.01, float(similarity + 1) / 2)  # Scale from [-1,1] to [0.01,1]
            else:
                return 0.1  # Default small weight
                
        except Exception as e:
            print(f"Warning: Could not compute chunk similarity using embeddings: {e}")
            return 0.1  # Default small weight
    
    def _compute_weighted_embedding(self, text: str, title: str) -> np.ndarray:
        """Compute weighted average embedding for text based on title similarity"""
        if not self.embedding_model:
            return np.zeros(384)  # Return zero vector if no model
        
        try:
            # Split text into chunks
            chunks = self._split_text_into_chunks(text, self.max_tokens)
            
            if not chunks:
                return np.zeros(384)
            
            # Compute embeddings for all chunks
            chunk_embeddings = self.embedding_model.encode(chunks)
            
            # If only one chunk, return its embedding
            if len(chunks) == 1:
                return chunk_embeddings[0]
            
            # Compute title similarity weights for each chunk
            weights = []
            for chunk in chunks:
                weight = self._compute_chunk_similarity_to_title(chunk, title)
                weights.append(weight)
            
            # Convert to numpy array and normalize
            weights = np.array(weights)
            
            # If all weights are zero, use uniform weights
            if np.sum(weights) == 0:
                weights = np.ones(len(chunks))
            
            # Normalize weights
            weights = weights / np.sum(weights)
            
            # Compute weighted average embedding
            weighted_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
            
            return weighted_embedding
            
        except Exception as e:
            print(f"Warning: Weighted embedding computation failed: {e}")
            # Fallback: just encode the truncated text
            try:
                truncated_text = text[:self.max_tokens * 4]  # Rough character limit
                return self.embedding_model.encode([truncated_text])[0]
            except:
                return np.zeros(384)
    
    def _create_query_from_job(self, persona: str, job_to_be_done: str) -> str:
        """Create search query from persona and job description using NLTK"""
        query_parts = []
        
        if NLTK_AVAILABLE:
            try:
                # Use NLTK for better tokenization
                persona_tokens = word_tokenize(persona.lower())
                job_tokens = word_tokenize(job_to_be_done.lower())
                
                # Filter and process tokens
                persona_filtered = [token for token in persona_tokens 
                                  if token.isalpha() and token not in STOP_WORDS]
                job_filtered = [token for token in job_tokens 
                              if token.isalpha() and token not in STOP_WORDS]
                
                # Apply stemming if available
                if STEMMER:
                    persona_filtered = [STEMMER.stem(token) for token in persona_filtered]
                    job_filtered = [STEMMER.stem(token) for token in job_filtered]
                
                # Combine and deduplicate
                all_terms = list(set(persona_filtered + job_filtered))
                return ' '.join(all_terms[:50])  # Limit query length 
                
            except Exception as e:
                print(f"NLTK query creation failed: {e}, falling back to regex")
        
        # Fallback to regex-based processing
        persona_terms = re.findall(r'\b\w+\b', persona.lower())
        job_terms = re.findall(r'\b\w+\b', job_to_be_done.lower())
        
        filtered_persona = [term for term in persona_terms if term not in STOP_WORDS and len(term) > 2]
        filtered_job = [term for term in job_terms if term not in STOP_WORDS and len(term) > 2]
        
        all_terms = list(set(filtered_persona + filtered_job))
        return ' '.join(all_terms[:15])

    
    def _calculate_length_penalty(self, text: str, title: str = None) -> float:
        """
        Calculate length penalty for a section based on content richness
        
        Args:
            text: Section content text
            title: Section title (optional, for context)
            
        Returns:
            Length penalty factor between 0.1 and 1.0
            - 1.0: Full score (good length)
            - 0.1: Heavy penalty (very short/empty content)
        """
        if not text or not text.strip():
            return 0.1  # Heavy penalty for empty content
        
        # Clean text for analysis
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Multiple length metrics
        char_count = len(clean_text)
        word_count = len(clean_text.split())
        
        # Sentence count using spacy (more robust)
        try:
            doc = nlp(clean_text)
            sentence_count = len([sent for sent in doc.sents if len(sent.text.strip()) > 5])
        except:
            # Fallback to simple sentence counting
            sentence_count = len([s for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 5])
        
        # Check for title-only sections (title repeated in content)
        is_title_only = False
        if title and title.strip():
            # Check if content is just the title or very similar
            title_clean = re.sub(r'\s+', ' ', title.strip().lower())
            content_clean = clean_text.lower()
            
            # Title-only detection
            if (content_clean == title_clean or 
                content_clean.startswith(title_clean) and len(content_clean) - len(title_clean) < 20):
                is_title_only = True
        
        # Heavy penalty for title-only sections
        if is_title_only:
            return 0.1
        
        # Define thresholds for different content types
        # Very short content (likely headers, titles, or fragments)
        if char_count < 50 or word_count < 10:
            return 0.1
        
        # Short content (minimal information)
        elif char_count < 150 or word_count < 25 or sentence_count < 2:
            return 0.3
        
        # Moderate content 
        elif char_count < 300 or word_count < 50 or sentence_count < 3:
            return 0.6
        
        # Good content length
        elif char_count < 800 or word_count < 150 or sentence_count < 8:
            return 0.9
        
        # Excellent content length
        else:
            return 1.0
    
    def _calculate_content_quality_score(self, text: str, title: str = None) -> float:
        """
        Calculate overall content quality score combining length and richness metrics
        
        Args:
            text: Section content text
            title: Section title (optional)
            
        Returns:
            Quality score between 0.1 and 1.0
        """
        if not text or not text.strip():
            return 0.1
        
        # Base length penalty
        length_penalty = self._calculate_length_penalty(text, title)
        
        # Additional quality metrics
        clean_text = re.sub(r'\s+', ' ', text.strip())
        
        # Information density metrics
        unique_word_ratio = 1.0
        if len(clean_text.split()) > 0:
            words = clean_text.lower().split()
            unique_words = set(words)
            unique_word_ratio = len(unique_words) / len(words)
        
        # Punctuation diversity (indicates structured content)
        punctuation_chars = set('.,;:!?()-[]{}')
        punct_diversity = len(set(clean_text) & punctuation_chars) / len(punctuation_chars)
        
        # Numeric content (often indicates data/statistics)
        has_numbers = bool(re.search(r'\d+', clean_text))
        number_bonus = 1.1 if has_numbers else 1.0
        
        # Combine metrics
        quality_multiplier = (
            0.7 * unique_word_ratio +  # Vocabulary diversity
            0.2 * punct_diversity +    # Structural complexity  
            0.1 * number_bonus         # Factual content bonus
        )
        
        # Ensure quality multiplier is reasonable
        quality_multiplier = max(0.5, min(1.2, quality_multiplier))
        
        # Final quality score
        final_score = length_penalty * quality_multiplier
        
        return max(0.1, min(1.0, final_score))
    
    def _stage1_bm25_filtering(self, sections: List[DocumentSection], query: str, number_docs) -> List[DocumentSection]:
        """Stage 1: Fast BM25 filtering to eliminate irrelevant sections"""

        top_k = max(5, number_docs//2)

        if not sections:
            return sections
        
        # Create corpus from section content and titles
        corpus = []
        for section in sections:
            # Combine title and content for better matching
            combined_text = f"{section.section_title} {section.content}"
            corpus.append(combined_text)
        
        # Initialize BM25
        bm25 = BM25(corpus)
        scores = bm25.get_scores(query)

        # normalize scores
        max_score = max(scores) if scores else 1
        scores = [score / max_score for score in scores]
        
        # Assign scores to sections
        for i, section in enumerate(sections):
            section.bm25_score = scores[i]
        
        # Sort by BM25 score and take top_k
        sections.sort(key=lambda x: x.bm25_score, reverse=True)
        return sections[:top_k]
    
    def _stage2_semantic_similarity(self, sections: List[DocumentSection], query: str) -> List[DocumentSection]:
        """Stage 2: Semantic embedding similarity for remaining candidates using chunked embeddings"""
        if not self.embedding_model or not sections:
            # Fallback: keep BM25 scores as semantic scores
            for section in sections:
                section.semantic_score = section.bm25_score
            return sections
        
        try:
            # Compute query embedding (query is typically short, so no chunking needed)
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Compute weighted embeddings for each section
            section_embeddings = []
            for section in sections:
                combined_text = f"{section.section_title} {section.content}"
                weighted_embedding = self._compute_weighted_embedding(combined_text, section.section_title)
                section_embeddings.append(weighted_embedding)
            
            # Calculate cosine similarities
            query_norm = np.linalg.norm(query_embedding)
            for i, section in enumerate(sections):
                section_norm = np.linalg.norm(section_embeddings[i])
                if query_norm > 0 and section_norm > 0:
                    similarity = np.dot(query_embedding, section_embeddings[i]) / (query_norm * section_norm)
                    section.semantic_score = float(similarity)
                else:
                    section.semantic_score = 0.0
        
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            # Fallback to BM25 scores
            max_bm25 = max(s.bm25_score for s in sections) if sections else 1
            for section in sections:
                section.semantic_score = section.bm25_score / max_bm25
        
        return sections
    
    def _named_entity_density(self, text):
        doc = nlp(text)
        num_entities = len(doc.ents)
        num_tokens = len([token for token in doc if not token.is_space])
        
        if num_tokens == 0:
            return 0.0  # Avoid division by zero
        if num_tokens < 20:
            return 0.1
        
        return num_entities / num_tokens
    
    def _stage3_score_ranking(self, sections: List[DocumentSection], persona: str, job_to_be_done: str) -> List[DocumentSection]:
        """Stage 3: Persona-aware re-ranking using job-specific weights with length penalties"""
        if not sections:
            return sections
        
        for section in sections:
            combined_text = f"{section.section_title} {section.content}"
            
            # Calculate existing metrics
            section.named_entity_density = self._named_entity_density(combined_text)
            
            # Calculate content quality score (includes length penalty)
            section.content_quality_score = self._calculate_content_quality_score(
                section.content, 
                section.section_title
            )
            
            # Combine all scores with weights, applying content quality penalty
            base_score = (
                0.15 * section.bm25_score +
                0.6 * section.semantic_score +
                0.25 * section.named_entity_density
            )
            
            # Apply content quality penalty
            section.final_score = base_score * section.content_quality_score
        
        # Sort by final score
        sections.sort(key=lambda x: x.final_score, reverse=True)
        return sections
    
    def _select_diverse_sections(self, sections: List[DocumentSection], max_sections: int = 5) -> List[DocumentSection]:
        """Select diverse sections to avoid redundancy"""
        if len(sections) <= max_sections:
            return sections
        
        selected = []
        document_counts = {}
        
        for section in sections:
            # Limit sections per document for diversity
            doc_count = document_counts.get(section.document, 0)
            if doc_count < 3:  # Max 3 sections per document
                selected.append(section)
                document_counts[section.document] = doc_count + 1
                
                if len(selected) >= max_sections:
                    break
        
        # If we still need more sections, add remaining highest-scored ones
        if len(selected) < max_sections:
            remaining = [s for s in sections if s not in selected]
            selected.extend(remaining[:max_sections - len(selected)])
        
        return selected
    
    def _refine_section_text(self, section: DocumentSection) -> str:
        """Refine and truncate section text for output"""
        text = section.content.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def _create_sliding_window_chunks(self, text: str, window_size: int = 3, overlap: int = 1) -> List[Dict[str, Any]]:
        """
        Create sliding window chunks from text using sentence tokenization
        
        Args:
            text: Input text to chunk
            window_size: Number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and sentence positions
        """
        if not text.strip():
            return []
        
        # Tokenize into sentences using spacy
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if len(sentences) == 0:
            return []
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(sentences):
            end_idx = min(start_idx + window_size, len(sentences))
            
            # Create chunk from sentences
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_text = ' '.join(chunk_sentences)
            
            chunk_info = {
                'text': chunk_text,
                'start_sentence': start_idx,
                'end_sentence': end_idx - 1,
                'num_sentences': len(chunk_sentences)
            }
            
            chunks.append(chunk_info)
            
            # Move start position considering overlap
            if end_idx >= len(sentences):
                break
            start_idx += max(1, window_size - overlap)
        
        return chunks
    
    def _find_relevant_subsections(self, documents: List[str], all_sections: List[Dict[str, Any]], 
                                 query: str, max_subsections: int = 5, window_size: int = 3, 
                                 overlap: int = 1) -> List[Dict[str, Any]]:
        """
        Find most relevant subsections using sliding window chunks with BM25 + semantic filtering
        
        Args:
            documents: List of document names to search through
            all_sections: All available sections from document parsing
            query: Search query (persona + job_to_be_done)
            max_subsections: Maximum number of subsections to return
            window_size: Number of sentences per sliding window chunk
            overlap: Number of sentences to overlap between chunks
            
        Returns:
            List of most relevant subsections with metadata
        """
        # Filter sections to only include those from selected documents
        candidate_sections = [s for s in all_sections if s['document'] in documents]
        
        if not candidate_sections:
            return []
        
        # print(f"Creating sliding window chunks from {len(candidate_sections)} sections in {len(documents)} documents...")
        
        # Create all sliding window chunks from candidate sections
        all_chunks = []
        
        for section_data in candidate_sections:
            section_text = section_data['content']
            chunks = self._create_sliding_window_chunks(section_text, window_size, overlap)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk_with_metadata = {
                    'document': section_data['document'],
                    'section_title': section_data['section_title'],
                    'page_number': section_data['page_number'],
                    'chunk_text': chunk['text'],
                    'start_sentence': chunk['start_sentence'],
                    'end_sentence': chunk['end_sentence'],
                    'num_sentences': chunk['num_sentences'],
                    'bm25_score': 0.0,
                    'semantic_score': 0.0
                }
                all_chunks.append(chunk_with_metadata)
        
        # print(f"Created {len(all_chunks)} sliding window chunks")
        
        if not all_chunks:
            return []
        
        # Stage 1: BM25 filtering on chunks
        # print("Stage 1: BM25 filtering on chunks...")
        chunk_corpus = [chunk['chunk_text'] for chunk in all_chunks]
        
        try:
            bm25 = BM25(chunk_corpus)
            bm25_scores = bm25.get_scores(query)
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_scores) if bm25_scores else 1
            normalized_bm25_scores = [score / max_bm25 for score in bm25_scores]
            
            # Assign BM25 scores to chunks
            for i, chunk in enumerate(all_chunks):
                chunk['bm25_score'] = normalized_bm25_scores[i]
            
            # Sort by BM25 score and take top candidates for semantic analysis
            all_chunks.sort(key=lambda x: x['bm25_score'], reverse=True)
            top_bm25_chunks = all_chunks[:min(50, len(all_chunks))]  # Top 50 for semantic analysis
            
        except Exception as e:
            print(f"Warning: BM25 filtering failed: {e}")
            top_bm25_chunks = all_chunks  # Use all chunks if BM25 fails
        
        # print(f"After BM25 filtering: {len(top_bm25_chunks)} chunks")
        
        # Stage 2: Semantic similarity on filtered chunks
        # print("Stage 2: Semantic similarity analysis on chunks...")
        
        if not self.embedding_model:
            print("Warning: No embedding model available, using BM25 scores only")
            # Use BM25 scores as semantic scores
            for chunk in top_bm25_chunks:
                chunk['semantic_score'] = chunk['bm25_score']
            final_chunks = top_bm25_chunks
        else:
            try:
                # Compute query embedding
                query_embedding = self.embedding_model.encode([query])[0]
                
                # Compute embeddings for chunk texts
                chunk_texts = [chunk['chunk_text'] for chunk in top_bm25_chunks]
                chunk_embeddings = self.embedding_model.encode(chunk_texts)
                
                # Calculate cosine similarities
                query_norm = np.linalg.norm(query_embedding)
                
                for i, chunk in enumerate(top_bm25_chunks):
                    chunk_norm = np.linalg.norm(chunk_embeddings[i])
                    
                    if query_norm > 0 and chunk_norm > 0:
                        similarity = np.dot(query_embedding, chunk_embeddings[i]) / (query_norm * chunk_norm)
                        chunk['semantic_score'] = float(similarity)
                    else:
                        chunk['semantic_score'] = 0.0
                
                final_chunks = top_bm25_chunks
                
            except Exception as e:
                print(f"Warning: Semantic similarity calculation failed: {e}")
                # Fallback to BM25 scores
                for chunk in top_bm25_chunks:
                    chunk['semantic_score'] = chunk['bm25_score']
                final_chunks = top_bm25_chunks
        
        # Stage 3: Final scoring and selection with length penalties
        # print("Stage 3: Final scoring with length penalties and diverse selection...")
        
        # Calculate content quality scores and combine with other scores
        for chunk in final_chunks:
            # Calculate content quality score for the chunk
            chunk['content_quality_score'] = self._calculate_content_quality_score(
                chunk['chunk_text'], 
                chunk['section_title']
            )
            
            # Base score combination
            base_score = 0.3 * chunk['bm25_score'] + 0.7 * chunk['semantic_score']
            
            # Apply content quality penalty
            chunk['final_score'] = base_score * chunk['content_quality_score']
        
        # Sort by final score
        final_chunks.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Select diverse top chunks (limit per document/section)
        selected_subsections = []
        document_section_counts = {}
        
        for chunk in final_chunks:
            doc_section_key = f"{chunk['document']}_{chunk['section_title']}"
            count = document_section_counts.get(doc_section_key, 0)
            
            # Limit chunks per document-section combination for diversity
            if count < 1:  # Max 2 chunks per section
                # Create subsection entry
                subsection_entry = {
                    "document": chunk['document'],
                    "section_title": chunk['section_title'],
                    "refined_text": chunk['chunk_text'],
                    "page_number": chunk['page_number'],
                    "sentence_range": f"{chunk['start_sentence']}-{chunk['end_sentence']}",
                    "num_sentences": chunk['num_sentences'],
                    "bm25_score": chunk['bm25_score'],
                    "semantic_score": chunk['semantic_score'],
                    "content_quality_score": chunk['content_quality_score'],
                    "final_score": chunk['final_score']
                }
                
                selected_subsections.append(subsection_entry)
                document_section_counts[doc_section_key] = count + 1
                
                if len(selected_subsections) >= max_subsections:
                    break
        
        # If we still need more subsections, add remaining highest-scored ones
        if len(selected_subsections) < max_subsections:
            remaining_chunks = [chunk for chunk in final_chunks 
                              if not any(s['refined_text'] == chunk['chunk_text'] 
                                       for s in selected_subsections)]
            
            for chunk in remaining_chunks[:max_subsections - len(selected_subsections)]:
                subsection_entry = {
                    "document": chunk['document'],
                    "section_title": chunk['section_title'],
                    "refined_text": chunk['chunk_text'],
                    "page_number": chunk['page_number'],
                    "sentence_range": f"{chunk['start_sentence']}-{chunk['end_sentence']}",
                    "num_sentences": chunk['num_sentences'],
                    "bm25_score": chunk['bm25_score'],
                    "semantic_score": chunk['semantic_score'],
                    "content_quality_score": chunk.get('content_quality_score', 1.0),
                    "final_score": chunk['final_score']
                }
                selected_subsections.append(subsection_entry)
        
        # print(f"Selected {len(selected_subsections)} most relevant subsections")
        return selected_subsections
    
    def analyze_documents(self, 
                         document_sections: List[Dict[str, Any]], 
                         persona: str, 
                         job_to_be_done: str,
                         input_documents: List[str]) -> Dict[str, Any]:
        """
        Main analysis function
        
        Args:
            document_sections: List of dictionaries with keys: 'document', 'section_title', 'content', 'page_number'
            persona: Persona description
            job_to_be_done: Job description
            input_documents: List of input document names
        
        Returns:
            Dictionary in the specified JSON format
        """
        start_time = datetime.now()
        
        # Store original sections for subsection analysis
        original_sections = document_sections.copy()
        
        # Convert input to DocumentSection objects
        sections = []
        for section_data in document_sections:
            section = DocumentSection(
                document=section_data['document'],
                section_title=section_data['section_title'],
                content=section_data['content'],
                page_number=section_data['page_number']
            )
            sections.append(section)
        
        # Create query from persona and job
        query = self._create_query_from_job(persona, job_to_be_done)
        full_query = f"{persona} {job_to_be_done}"  # Full query for subsection analysis
        
        # Stage 1: BM25 filtering
        # print(f"Stage 1: BM25 filtering from {len(sections)} sections...")
        filtered_sections = self._stage1_bm25_filtering(sections, query, number_docs=len(input_documents))
        # print(f"After BM25: {len(filtered_sections)} sections")
        
        # Stage 2: Semantic similarity with chunked embeddings
        # print("Stage 2: Semantic similarity analysis with chunked embeddings...")
        semantic_sections = self._stage2_semantic_similarity(filtered_sections, query)
        
        # Stage 3: Persona-aware re-ranking
        # print("Stage 3: Persona-aware re-ranking...")
        final_sections = self._stage3_score_ranking(semantic_sections, persona, job_to_be_done)
        
        # Select diverse top sections
        top_sections = self._select_diverse_sections(final_sections, max_sections=5)
        
        # Assign importance ranks
        for i, section in enumerate(top_sections):
            section.importance_rank = i + 1
        
        # Extract unique documents from top sections
        selected_documents = list(set(section.document for section in top_sections))
        # print(f"Selected documents for subsection analysis: {selected_documents}")
        
        # Stage 4: Find most relevant subsections from selected documents
        # print("Stage 4: Finding most relevant subsections using sliding window chunks...")
        relevant_subsections = self._find_relevant_subsections(
            documents=selected_documents,
            all_sections=original_sections,
            query=full_query,
            max_subsections=5,  # Default parameter
            window_size=4,      # Default: 3 sentences per chunk
            overlap=2           # Default: 1 sentence overlap
        )
        
        # Create output format
        output = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": start_time.isoformat(),
                # "max_tokens_per_chunk": self.max_tokens,
                # "selected_documents_for_subsection_analysis": selected_documents,
                # "num_subsections_analyzed": len(relevant_subsections)
            },
            "extracted_sections": [
                {
                    "document": section.document,
                    "section_title": section.section_title,
                    "importance_rank": section.importance_rank,
                    "page_number": section.page_number + 1,  # Convert to 1-based index
                    # remove later
                    # "named_entity_density": section.named_entity_density,
                    # "content_quality_score": section.content_quality_score,
                    # "bm25_score": section.bm25_score,
                    # "semantic_score": section.semantic_score,
                    # "final_score": section.final_score
                }
                for section in top_sections
            ],
            "subsection_analysis": [
                {
                    "document": subsection["document"],
                    "refined_text": subsection["refined_text"],
                    "page_number": subsection["page_number"] + 1,  # Convert to 1-based index
                }
                for subsection in relevant_subsections
            ]
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        # print(f"Processing completed in {processing_time:.2f} seconds")
        
        return output

def chunk_document(doc, document_text, headings, pdf_file):
    # Extract text of each page once
    page_texts = [page.get_text() for page in doc]

    # Map heading text to the page number where it appears
    heading_pages = {}
    for heading in headings:
        h_text = heading["text"]
        page_num = None
        for pg_num, text in enumerate(page_texts):
            if h_text in text:
                page_num = pg_num
                break
        heading_pages[h_text] = page_num

    chunks = []
    if headings:
        # Intro chunk before the first heading
        first_h = headings[0]["text"]
        first_pos = document_text.find(first_h)
        if first_pos > 0:
            intro = document_text[:first_pos].strip()
            if intro:
                chunks.append({
                    "pdf_file": pdf_file,
                    "page_num": 0,
                    "heading": None,
                    "content": intro
                })

        # Chunks under each heading (exclude heading from content)
        for i, heading in enumerate(headings):
            h_text = heading["text"]
            start = document_text.find(h_text)
            if start == -1:
                continue  # Skip if heading not found
            start += len(h_text)  # Move start past the heading text
            if i + 1 < len(headings):
                next_h_text = headings[i + 1]["text"]
                end = document_text.find(next_h_text)
                if end == -1:
                    end = len(document_text)
            else:
                end = len(document_text)
            chunk_text = document_text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "pdf_file": pdf_file,
                    "page_num": heading_pages.get(h_text),
                    "heading": h_text,
                    "content": chunk_text
                })
    else:
        # No headings: single chunk
        content = document_text.strip()
        if content:
            chunks.append({
                "pdf_file": pdf_file,
                "page_num": 0,
                "heading": None,
                "content": content
            })

    return chunks



def get_sections(dir_path) -> List[DocumentSection]:
    """Create sections from pdf files."""
    file_names = os.listdir(dir_path)
    pdf_files = [f for f in file_names if f.lower().endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the directory.")
        return []

    all_chunks = []

    for pdf_file in pdf_files:
        pdf_path = os.path.join(dir_path, pdf_file)
        document_outline = run_parser(pdf_path)

        # Read full text & open doc for page lookup
        doc = fitz.open(pdf_path)
        document_text = "".join(page.get_text() for page in doc)
        headings = document_outline.get("headings", [])

        # Chunk and record
        chunks = chunk_document(doc, document_text, headings, pdf_file)
        all_chunks.extend(chunks)
        doc.close()
    
    # with open("all_chunks.json", "w", encoding="utf-8") as f:
    #     json.dump(all_chunks, f, indent=4)
    
    # Convert all_chunks to DocumentSection objects
    sections = []
    for chunk in all_chunks:
        section = {
            "document": chunk["pdf_file"],
            "section_title": chunk["heading"] if chunk["heading"] else "Introduction",
            "content": chunk["content"],
            "page_number": chunk["page_num"] if chunk["page_num"] is not None else 0
        }
        sections.append(section)
    
    # print(f"Successfully created {len(sections)} sections from {len(all_chunks)} chunks")
    return sections


def usage(dir_path, persona, job_to_be_done):

    start_time = time.time()
    print(f"Starting analysis for persona: {persona}, job: {job_to_be_done}")
    sections = get_sections(dir_path)
    pdf_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.pdf')]
    
    # Initialize the analyst with max_tokens parameter
    analyst = IntelligentDocumentAnalyst(max_tokens=512)
    
    # Analyze documents
    result = analyst.analyze_documents(
        document_sections=sections,
        persona=persona,
        job_to_be_done=job_to_be_done,
        input_documents=pdf_files
    )

    # save result to a json file
    with open(f"/app/output/result_{persona}_{job_to_be_done}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    
    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds for {len(pdf_files)} PDF files")

    return

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python output_1b.py <persona> <job_to_be_done>")
        sys.exit(1)

    persona = sys.argv[1]
    job_to_be_done = sys.argv[2]
    pdf_dir = "/app/input"

    usage(pdf_dir, persona, job_to_be_done)