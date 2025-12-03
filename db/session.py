# db/session.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Создаем базовый класс для моделей
Base = declarative_base()

# Получаем URL из настроек
DATABASE_URL = settings.DATABASE_URL

# Для SQLite используем aiosqlite
if "sqlite" in DATABASE_URL:
    # Заменяем драйвер на aiosqlite для async работы
    DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    logger.info(f"Using SQLite database: {DATABASE_URL}")

    engine = create_async_engine(
        DATABASE_URL,
        echo=settings.DEBUG,
        poolclass=NullPool,
        connect_args={"check_same_thread": False}
    )
# Для PostgreSQL используем asyncpg
elif "postgresql" in DATABASE_URL:
    logger.info(f"Using PostgreSQL database: {DATABASE_URL}")

    engine = create_async_engine(
        DATABASE_URL,
        echo=settings.DEBUG,
        pool_size=20,
        max_overflow=30,
        pool_pre_ping=True
    )
else:
    raise ValueError(f"Unsupported database URL: {DATABASE_URL}")

# Создаем фабрику сессий
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def init_db():
    """Initialize database (create tables)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized")


async def close_db():
    """Close database connections"""
    await engine.dispose()
    logger.info("Database connections closed")


# Для обратной совместимости с импортами
get_session = AsyncSessionLocal
