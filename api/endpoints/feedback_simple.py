# api/endpoints/feedback_simple.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

router = APIRouter()


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int  # 1-5
    comment: Optional[str] = None
    suggestion: Optional[str] = None


# In-memory storage
feedbacks = []


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Отправить обратную связь"""
    if not 1 <= feedback.rating <= 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")

    feedback_data = {
        "feedback_id": f"fb_{int(datetime.now().timestamp())}",
        "message_id": feedback.message_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "suggestion": feedback.suggestion,
        "timestamp": datetime.now().isoformat()
    }

    feedbacks.append(feedback_data)

    return {
        "status": "received",
        **feedback_data,
        "total_feedbacks": len(feedbacks)
    }


@router.get("/feedback/stats")
async def get_feedback_stats():
    """Статистика по обратной связи"""
    if not feedbacks:
        return {
            "total": 0,
            "average_rating": 0,
            "ratings_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
        }

    ratings = [f["rating"] for f in feedbacks]
    distribution = {str(i): 0 for i in range(1, 6)}
    for rating in ratings:
        distribution[str(rating)] += 1

    return {
        "total": len(feedbacks),
        "average_rating": sum(ratings) / len(ratings),
        "ratings_distribution": distribution
    }