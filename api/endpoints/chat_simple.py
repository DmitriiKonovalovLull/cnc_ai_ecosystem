# api/endpoints/chat_simple.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

router = APIRouter()


class ChatMessage(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    message_id: str
    user_message: str
    ai_response: str
    conversation_id: str
    timestamp: str
    confidence: float


# In-memory storage for demo
conversations = {}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatMessage):
    """Отправить сообщение в чат"""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Генерируем ID сообщения
    message_id = f"msg_{int(datetime.now().timestamp())}"

    # Создаем или получаем conversation_id
    conversation_id = request.conversation_id or f"conv_{request.user_id or 'anonymous'}"

    # Простой AI ответ (заглушка)
    ai_response = f"""Я получил ваше сообщение: "{request.message}"

В настоящее время AI модель для обработки сообщений находится в разработке.

Доступные функции в будущем:
1. Ответы на вопросы по CNC станкам
2. Помощь с настройкой параметров
3. Диагностика проблем
4. Рекомендации по инструментам

Ваше сообщение сохранено для обучения модели."""

    # Сохраняем в памяти
    if conversation_id not in conversations:
        conversations[conversation_id] = []

    conversations[conversation_id].append({
        "message_id": message_id,
        "user_message": request.message,
        "ai_response": ai_response,
        "timestamp": datetime.now().isoformat()
    })

    return ChatResponse(
        message_id=message_id,
        user_message=request.message,
        ai_response=ai_response,
        conversation_id=conversation_id,
        timestamp=datetime.now().isoformat(),
        confidence=0.8
    )


@router.get("/chat/history")
async def get_chat_history(conversation_id: Optional[str] = None, limit: int = 20):
    """Получить историю чата"""
    if conversation_id:
        history = conversations.get(conversation_id, [])
    else:
        # Все разговоры (для демо)
        history = []
        for conv in conversations.values():
            history.extend(conv[-5:])  # Последние 5 сообщений из каждого разговора

    history = sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]

    return {
        "conversation_id": conversation_id or "all",
        "history": history,
        "count": len(history),
        "total_conversations": len(conversations)
    }


@router.delete("/chat/conversation")
async def delete_conversation(conversation_id: str):
    """Удалить разговор"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"status": "deleted", "conversation_id": conversation_id}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")