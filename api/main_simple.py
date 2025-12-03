# api/main_simple.py
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="CNC AI API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "API работает!", "status": "OK"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/test")
async def test():
    return {"test": "success", "message": "Все системы работают"}

if __name__ == "__main__":
    # Исправляем запуск - указываем объект app, а не строку
    uvicorn.run(
        app,  # ← передаем объект приложения
        host="0.0.0.0",
        port=8000,
        reload=True
    )