import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple, Dict
import os
from dotenv import load_dotenv
import time
import logging

load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AgenticRAG:
    """Agentic RAG system with multi-step reasoning."""
    
    def __init__(self, embedding_model: SentenceTransformer):
        """
        Initialize agentic RAG.
        
        Args:
            embedding_model: Pre-loaded sentence transformer model
        """
        self.embedding_model = embedding_model
        
        # Configure Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.max_retries = 3
        self.retry_delay = 1  # seconds
    
    def _call_gemini_with_retry(self, prompt: str, timeout: int = 30) -> str:
        """
        Call Gemini API with retry logic and timeout.
        
        Args:
            prompt: The prompt to send
            timeout: Timeout in seconds
            
        Returns:
            API response text or error message
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"📡 Gemini API call (attempt {attempt + 1}/{self.max_retries})")
                response = self.model.generate_content(prompt, request_options={"timeout": timeout})
                
                if response and response.text:
                    logger.info("✅ API call successful")
                    return response.text
                else:
                    raise ValueError("Empty response from API")
                    
            except (TimeoutError, ConnectionError) as e:
                logger.warning(f"⏱️ Timeout/Connection error (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # exponential backoff
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return f"❌ API timeout after {self.max_retries} attempts. Please try again."
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    logger.warning(f"🚫 Rate limited (attempt {attempt + 1}): {error_msg}")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)
                        logger.info(f"⏳ Rate limit: waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        return "❌ API rate limit exceeded. Please wait a moment and try again."
                else:
                    logger.error(f"❌ API error: {error_msg}")
                    return f"❌ API error: {error_msg}"
        
        return "❌ API call failed after all retries."
    
    def _evaluate_retrieval_quality(self, question: str, chunks: List[str]) -> Dict:
        """
        Agent evaluates if retrieved chunks are sufficient.
        
        Returns:
            {
                "sufficient": bool,
                "confidence": float,
                "reasoning": str,
                "suggested_refinement": str or None
            }
        """
        # Validate input
        if not chunks or len(chunks) == 0:
            logger.warning("⚠️ No chunks available for evaluation")
            return {
                "sufficient": False,
                "confidence": 0.0,
                "reasoning": "No relevant content found in book.",
                "suggested_refinement": question + " (different source needed)"
            }
        
        evaluation_prompt = f"""You are an AI agent evaluating search results quality.

Question: {question}

Retrieved chunks (first 3):
1. {chunks[0][:200]}...
2. {chunks[1][:200] if len(chunks) > 1 else "N/A"}...
3. {chunks[2][:200] if len(chunks) > 2 else "N/A"}...

Evaluate:
1. Do these chunks contain relevant information to answer the question?
2. Is more information needed?
3. What specific aspect is missing (if any)?

Respond in this exact format:
SUFFICIENT: yes/no
CONFIDENCE: 0.0-1.0
REASONING: brief explanation
REFINEMENT: suggested refined query (or "none")
"""
        
        try:
            text = self._call_gemini_with_retry(evaluation_prompt)
            
            # Check if error occurred
            if text.startswith("❌"):
                logger.warning(f"Quality evaluation failed: {text}")
                return {
                    "sufficient": True,
                    "confidence": 0.6,
                    "reasoning": "Evaluation service temporary issue, proceeding with caution",
                    "suggested_refinement": None
                }
            
            # Parse response
            sufficient = "yes" in text.lower().split("sufficient:")[1].split("\n")[0] if "sufficient:" in text.lower() else False
            
            # Extract confidence
            confidence = 0.5
            try:
                for word in text.split():
                    try:
                        val = float(word)
                        if 0 <= val <= 1:
                            confidence = val
                            break
                    except:
                        pass
            except:
                confidence = 0.5
            
            return {
                "sufficient": sufficient,
                "confidence": min(confidence, 1.0),
                "reasoning": text,
                "suggested_refinement": None if sufficient else question + " detailed explanation"
            }
        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            # Fallback: assume sufficient
            return {
                "sufficient": True,
                "confidence": 0.5,
                "reasoning": "Evaluation skipped due to error, using default",
                "suggested_refinement": None
            }
    
    def retrieve_with_reasoning(
        self,
        question: str,
        index: faiss.Index,
        chunks: List[str],
        max_iterations: int = 2,
        top_k: int = 5
    ) -> Tuple[List[str], Dict]:
        """
        Agentic retrieval with multi-step reasoning.
        
        Args:
            question: User's question
            index: FAISS index
            chunks: Text chunks
            max_iterations: Max refinement steps
            top_k: Chunks per iteration
            
        Returns:
            (retrieved_chunks, agent_log)
        """
        agent_log = {
            "iterations": [],
            "total_chunks_retrieved": 0,
            "final_confidence": 0.0
        }
        
        all_retrieved_chunks = []
        current_query = question
        
        for iteration in range(max_iterations):
            print(f"\n🤖 Agent Iteration {iteration + 1}/{max_iterations}")
            print(f"   Query: {current_query}")
            
            # Retrieve chunks
            query_embedding = self.embedding_model.encode([current_query], convert_to_numpy=True)
            distances, indices = index.search(query_embedding.astype('float32'), top_k)
            
            iteration_chunks = [chunks[idx] for idx in indices[0]]
            all_retrieved_chunks.extend(iteration_chunks)
            
            print(f"   📄 Retrieved {len(iteration_chunks)} chunks")
            
            # Evaluate quality
            evaluation = self._evaluate_retrieval_quality(question, iteration_chunks)
            
            agent_log["iterations"].append({
                "iteration": iteration + 1,
                "query": current_query,
                "chunks_retrieved": len(iteration_chunks),
                "evaluation": evaluation
            })
            
            print(f"   ✅ Sufficient: {evaluation['sufficient']}")
            print(f"   📊 Confidence: {evaluation['confidence']:.2f}")
            
            # Check if we should continue
            if evaluation["sufficient"] or iteration == max_iterations - 1:
                agent_log["final_confidence"] = evaluation["confidence"]
                break
            
            # Refine query for next iteration
            if evaluation["suggested_refinement"]:
                current_query = evaluation["suggested_refinement"]
                print(f"   🔄 Refining query...")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_chunks = []
        for chunk in all_retrieved_chunks:
            if chunk not in seen:
                seen.add(chunk)
                unique_chunks.append(chunk)
        
        agent_log["total_chunks_retrieved"] = len(unique_chunks)
        
        print(f"\n✅ Agentic retrieval complete")
        print(f"   Total unique chunks: {len(unique_chunks)}")
        print(f"   Final confidence: {agent_log['final_confidence']:.2f}")
        
        return unique_chunks[:10], agent_log  # Return max 10 chunks
    
    def generate_answer(
        self,
        question: str,
        chunks: List[str],
        difficulty: str,
        agent_log: Dict
    ) -> str:
        """Generate answer using retrieved chunks with validation."""
        
        # Validate inputs
        if not question or not question.strip():
            return "❌ Error: Question is empty"
        
        if not chunks or len(chunks) == 0:
            return "❌ Error: No relevant content found in the selected book. Please try:\n1. Different search terms\n2. Select a different book\n3. Check if book content is properly indexed"
        
        # Create context with validation
        valid_chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
        if not valid_chunks:
            return "❌ Error: Retrieved content is empty. Please try again."
        
        context = "\n\n".join([f"[Source {i+1}]\n{chunk}" for i, chunk in enumerate(valid_chunks)])
        
        if len(context) == 0:
            return "❌ Error: No valid context to generate answer from."
        
        difficulty_instructions = {
            "beginner": "Explain in very simple terms with analogies and examples.",
            "intermediate": "Provide moderate technical detail with clear explanations.",
            "advanced": "Give detailed technical explanation with advanced concepts."
        }
        
        # Include agent reasoning in prompt
        agent_info = f"The system performed {len(agent_log['iterations'])} search iterations with {agent_log['final_confidence']:.0%} confidence."
        
        prompt = f"""You are an expert learning tutor.

{agent_info}

Student Level: {difficulty.upper()}
{difficulty_instructions.get(difficulty, difficulty_instructions["beginner"])}

Context from textbook ({len(valid_chunks)} chunks):
{context}

Student Question: {question}

Instructions:
- Answer based ONLY on the context above
- If context is insufficient, clearly state what's missing
- Explain step-by-step
- Use examples from the context
- Be encouraging and supportive

Answer:"""
        
        logger.info(f"🔄 Generating answer (difficulty: {difficulty}, chunks: {len(valid_chunks)})")
        response = self._call_gemini_with_retry(prompt, timeout=30)
        
        if response.startswith("❌"):
            logger.error(f"Answer generation failed: {response}")
        else:
            logger.info("✅ Answer generated successfully")
        
        return response