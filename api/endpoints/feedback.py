from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from api.dependencies import get_current_user
from knowledge_base.feedback_loop import FeedbackLoop

router = APIRouter()


# Request/Response Models
class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original user query")
    original_response: str = Field(..., description="Original AI response")
    corrected_response: str = Field(..., description="Corrected response from operator")
    operator_id: str = Field(..., description="Operator ID")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in correction")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str
    timestamp: str


class QualityFeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")
    comments: Optional[str] = None
    user_id: str


class BatchFeedbackRequest(BaseModel):
    items: List[FeedbackRequest] = Field(..., min_items=1, max_items=50)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
        request: FeedbackRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_current_user)
):
    """
    Submit feedback from operator to improve AI responses
    """
    try:
        # Get feedback loop from app state
        from api.main import app

        feedback_loop = app.state.feedback_loop

        # Submit feedback
        feedback_id = await feedback_loop.submit_feedback(
            query=request.query,
            original_response=request.original_response,
            corrected_response=request.corrected_response,
            operator_id=request.operator_id,
            confidence=request.confidence,
            metadata=request.metadata
        )

        # Trigger background processing
        background_tasks.add_task(
            process_feedback_background,
            feedback_loop,
            feedback_id
        )

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="received",
            message="Feedback submitted successfully",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting feedback: {str(e)}"
        )


@router.post("/feedback/batch", response_model=Dict[str, Any])
async def submit_batch_feedback(
        request: BatchFeedbackRequest,
        background_tasks: BackgroundTasks,
        current_user: dict = Depends(get_current_user)
):
    """
    Submit multiple feedback items in batch
    """
    try:
        from api.main import app
        feedback_loop = app.state.feedback_loop

        results = []

        for item in request.items:
            try:
                feedback_id = await feedback_loop.submit_feedback(
                    query=item.query,
                    original_response=item.original_response,
                    corrected_response=item.corrected_response,
                    operator_id=item.operator_id,
                    confidence=item.confidence,
                    metadata=item.metadata
                )

                results.append({
                    "feedback_id": feedback_id,
                    "status": "received",
                    "original_query": item.query[:50] + "..." if len(item.query) > 50 else item.query
                })

            except Exception as e:
                results.append({
                    "error": str(e),
                    "status": "failed",
                    "original_query": item.query[:50] + "..." if len(item.query) > 50 else item.query
                })

        # Trigger batch processing
        background_tasks.add_task(
            process_feedback_batch,
            feedback_loop,
            [r["feedback_id"] for r in results if "feedback_id" in r]
        )

        return {
            "total_submitted": len(request.items),
            "successful": sum(1 for r in results if r["status"] == "received"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting batch feedback: {str(e)}"
        )


@router.post("/feedback/quality")
async def submit_quality_feedback(
        request: QualityFeedbackRequest,
        current_user: dict = Depends(get_current_user)
):
    """
    Submit quality feedback for responses
    """
    try:
        # Log quality feedback
        quality_data = {
            "query": request.query,
            "response": request.response,
            "rating": request.rating,
            "comments": request.comments,
            "user_id": request.user_id,
            "timestamp": datetime.now().isoformat(),
            "type": "quality_feedback"
        }

        # Save to analytics
        from pathlib import Path
        import json

        log_file = Path("logs/quality_feedback.jsonl")
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(quality_data, ensure_ascii=False) + "\n")

        return {
            "status": "success",
            "message": "Quality feedback recorded",
            "rating": request.rating,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error submitting quality feedback: {str(e)}"
        )


@router.get("/feedback/stats")
async def get_feedback_stats(
        time_period: Optional[str] = "7d",
        current_user: dict = Depends(get_current_user)
):
    """
    Get feedback statistics
    """
    try:
        from api.main import app
        from datetime import timedelta

        feedback_loop = app.state.feedback_loop

        # Parse time period
        period_map = {
            "1d": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
            "90d": timedelta(days=90),
            "all": None
        }

        period = period_map.get(time_period, timedelta(days=7))

        stats = await feedback_loop.get_feedback_stats(period)

        return {
            "time_period": time_period,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting feedback stats: {str(e)}"
        )


@router.get("/feedback/training-data")
async def get_training_data(
        limit: Optional[int] = 100,
        min_confidence: Optional[float] = 0.7,
        current_user: dict = Depends(get_current_user)
):
    """
    Get processed feedback as training data
    """
    try:
        from api.main import app

        if not current_user.get("is_superuser", False):
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )

        feedback_loop = app.state.feedback_loop

        training_data = await feedback_loop.get_training_data(limit)

        # Filter by confidence
        filtered_data = [
            item for item in training_data
            if item.get('confidence', 0) >= min_confidence
        ]

        return {
            "total": len(training_data),
            "filtered": len(filtered_data),
            "min_confidence": min_confidence,
            "data": filtered_data[:limit],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting training data: {str(e)}"
        )


@router.delete("/feedback/cleanup")
async def cleanup_old_feedback(
        max_age_days: int = 90,
        current_user: dict = Depends(get_current_user)
):
    """
    Cleanup old feedback items
    """
    try:
        from api.main import app

        if not current_user.get("is_superuser", False):
            raise HTTPException(
                status_code=403,
                detail="Admin access required"
            )

        feedback_loop = app.state.feedback_loop

        result = await feedback_loop.cleanup_old_feedback(max_age_days)

        return {
            "action": "cleanup",
            "max_age_days": max_age_days,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(e)}"
        )


async def process_feedback_background(feedback_loop: FeedbackLoop, feedback_id: str):
    """
    Background task to process feedback
    """
    try:
        # Additional processing logic can be added here
        # For example, trigger model retraining

        from ml_core.training.trainer import ModelTrainer

        # Check if we have enough feedback for training
        stats = await feedback_loop.get_feedback_stats()

        if stats.get('processed', 0) >= 50:  # Threshold for retraining
            trainer = ModelTrainer()

            # Get training data
            training_data = await feedback_loop.get_training_data(limit=100)

            if training_data:
                # Convert to training format
                # This would depend on your actual training data format
                pass

    except Exception as e:
        print(f"Background feedback processing error: {e}")


async def process_feedback_batch(feedback_loop: FeedbackLoop, feedback_ids: List[str]):
    """
    Process batch of feedback items
    """
    try:
        # Batch processing logic
        # For example, update multiple models at once

        print(f"Processing batch of {len(feedback_ids)} feedback items")

    except Exception as e:
        print(f"Batch feedback processing error: {e}")