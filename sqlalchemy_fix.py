# sqlalchemy_fix.py
import sys
import warnings


class AsyncSessionStub:
    """Заглушка для AsyncSession"""

    def __init__(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, *args, **kwargs):
        return None

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        pass


class declarative_base_stub:
    """Заглушка для declarative_base"""

    @staticmethod
    def __call__():
        class Base:
            metadata = type('Metadata', (), {})()

        return Base


# Создаем фиктивные модули
class FakeModule:
    def __init__(self, name):
        self.__name__ = name
        sys.modules[name] = self

    def __getattr__(self, name):
        # Возвращаем заглушки для любых атрибутов
        if name == 'AsyncSession':
            return AsyncSessionStub
        elif name == 'declarative_base':
            return declarative_base_stub
        else:
            return type(name, (), {})


# Создаем фиктивные модули SQLAlchemy
FakeModule('sqlalchemy.ext.asyncio')
FakeModule('sqlalchemy.orm')