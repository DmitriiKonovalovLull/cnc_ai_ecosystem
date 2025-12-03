# Создай тестовый файл test_env.py в корне проекта:
import os
from dotenv import load_dotenv

load_dotenv()

print("DATABASE_URL:", os.getenv("DATABASE_URL"))
print("DEBUG:", os.getenv("DEBUG"))
print("JWT_SECRET_KEY exists:", bool(os.getenv("JWT_SECRET_KEY")))