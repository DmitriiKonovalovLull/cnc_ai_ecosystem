# Создай тестовый файл test_env.py в корне проекта:
import os
from dotenv import load_dotenv

load_dotenv()

print("DATABASE_URL configured:", bool(os.getenv("DATABASE_URL")))
print("DEBUG enabled:", os.getenv("DEBUG", "False").lower() == "true")
print("JWT_SECRET_KEY configured:", bool(os.getenv("JWT_SECRET_KEY")))
