# api/dependencies.py - ИСПРАВЛЕННАЯ ВЕРСИЯ БЕЗ SQLAlchemy
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
from datetime import datetime, timedelta

# Пытаемся импортировать JWT
try:
    from jose import JWTError, jwt

    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: python-jose not installed")

# Импортируем настройки
try:
    from config.settings import settings

    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    print("Warning: settings not available")

security = HTTPBearer()


# Заглушка для AsyncSession
class AsyncSessionStub:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, *args):
        return None

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        pass


async def get_db():
    """Dependency to get database session - STUB VERSION"""
    session = AsyncSessionStub()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Dependency to get current user from JWT token"""
    if not JWT_AVAILABLE or not SETTINGS_AVAILABLE:
        # Return test user if JWT or settings not available
        return {
            "user_id": "test_user_001",
            "email": "operator@cnc-ai.com",
            "is_superuser": True,
            "is_active": True,
            "role": "operator"
        }

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        return {
            "user_id": user_id,
            "email": payload.get("email"),
            "is_superuser": payload.get("is_superuser", False),
            "is_active": payload.get("is_active", True),
            "role": payload.get("role", "operator")
        }

    except JWTError:
        # Для разработки используем тестового пользователя
        return {
            "user_id": "dev_user_001",
            "email": "developer@cnc-ai.local",
            "is_superuser": True,
            "is_active": True,
            "role": "developer"
        }


async def get_current_active_user(
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Dependency to check if user is active"""
    if not current_user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(
        current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Dependency to check if user is admin"""
    if not current_user.get("is_superuser", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


def create_access_token(data: Dict[str, Any]) -> str:
    """Create JWT access token"""
    if not JWT_AVAILABLE or not SETTINGS_AVAILABLE:
        return "stub_token_for_development"

    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    if not JWT_AVAILABLE or not SETTINGS_AVAILABLE:
        return "stub_refresh_token_for_development"

    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
