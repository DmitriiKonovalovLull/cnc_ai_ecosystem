from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

from api.dependencies import get_current_user
from ml_core.inference.answer_engine import AnswerEngine
from knowledge_base.feedback_loop import FeedbackLoop

router = APIRouter()
answer_engine = AnswerEngine()
security = HTTPBearer()


# Request/Response Models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    language: str = Field("ru", description="Response language")


class ChatResponse(BaseModel):
    query: str
    answer: str
    session_id: str
    intent: Dict[str, Any]
    entities: List[Dict[str, Any]]
    parameters: Dict[str, Any]
    sources: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_actions: List[Dict[str, str]]
    processing_time: float
    timestamp: str
    needs_feedback: bool = False


class BatchChatRequest(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=10)
    session_id: Optional[str] = None
    language: str = "ru"


class BatchChatResponse(BaseModel):
    results: List[Dict[str, Any]]
    session_id: str
    total_processing_time: float
    timestamp: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
        request: ChatRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_current_user)
):
    """
    Process user chat query and generate AI response
    """
    try:
        start_time = datetime.now()

        # Process query
        result = answer_engine.process_query(
            query=request.query,
            context=request.context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Determine if feedback is needed
        needs_feedback = (
                result.get('confidence', 0) < 0.7 or  # Low confidence
                len(result.get('sources', [])) < 2 or  # Few sources
                "не знаю" in result.get('answer', '').lower()  # Uncertain answer
        )

        response = ChatResponse(
            query=request.query,
            answer=result['answer'],
            session_id=session_id,
            intent=result.get('intent', {}),
            entities=result.get('entities', []),
            parameters=result.get('parameters', {}),
            sources=result.get('sources', []),
            confidence=result.get('confidence', 0.0),
            suggested_actions=result.get('suggested_actions', []),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            needs_feedback=needs_feedback
        )

        # Log interaction for training
        background_tasks.add_task(
            log_interaction,
            query=request.query,
            response=response.dict(),
            user_id=current_user.get('user_id', 'anonymous')
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.post("/chat/batch", response_model=BatchChatResponse)
async def batch_chat(
        request: BatchChatRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_current_user)
):
    """
    Process multiple chat queries in batch
    """
    try:
        start_time = datetime.now()
        results = []

        for query in request.queries:
            try:
                result = answer_engine.process_query(query)
                results.append({
                    "query": query,
                    "answer": result.get('answer', ''),
                    "intent": result.get('intent', {}),
                    "confidence": result.get('confidence', 0.0),
                    "entities": result.get('entities', []),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                })

                # Log each interaction
                background_tasks.add_task(
                    log_interaction,
                    query=query,
                    response=result,
                    user_id=current_user.get('user_id', 'anonymous')
                )

            except Exception as e:
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })

        total_time = (datetime.now() - start_time).total_seconds()
        session_id = request.session_id or str(uuid.uuid4())

        return BatchChatResponse(
            results=results,
            session_id=session_id,
            total_processing_time=total_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing batch: {str(e)}"
        )


@router.get("/chat/history")
async def get_chat_history(
        session_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        current_user: dict = Depends(get_current_user)
):
    """
    Get chat history for user or session
    """
    try:
        # This would typically query a database
        # For now, return mock data
        return {
            "user_id": current_user.get('user_id'),
            "session_id": session_id,
            "history": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching history: {str(e)}"
        )


@router.delete("/chat/history")
async def clear_chat_history(
        session_id: Optional[str] = None,
        current_user: dict = Depends(get_current_user)
):
    """
    Clear chat history for user or session
    """
    try:
        # This would typically delete from database
        return {
            "success": True,
            "message": "History cleared",
            "user_id": current_user.get('user_id'),
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing history: {str(e)}"
        )


@router.post("/chat/search")
async def search_knowledge(
        query: str,
        limit: int = 10,
        current_user: dict = Depends(get_current_user)
):
    """
    Search knowledge base directly
    """
    try:
        results = answer_engine.search_similar(query, limit)

        return {
            "query": query,
            "results": results,
            "total": len(results),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


async def log_interaction(query: str, response: Dict[str, Any], user_id: str):
    """
    Log chat interaction for training and analytics
    """
    try:
        from knowledge_base.kb_manager import KnowledgeBaseManager

        interaction_data = {
            "query": query,
            "response": response,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "type": "chat_interaction"
        }

        # Save to training data
        kb_manager = KnowledgeBaseManager()

        # You would save this to a database or file
        # For now, just log
        import json
        log_entry = json.dumps(interaction_data, ensure_ascii=False)

        from pathlib import Path
        log_file = Path("logs/chat_interactions.jsonl")
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    except Exception as e:
        print(f"Error logging interaction: {e}")
