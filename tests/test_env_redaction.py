import contextlib
import io
import os
import runpy
import sys
import types
import unittest
from unittest.mock import patch


class EnvironmentDiagnosticTests(unittest.TestCase):
    def test_diagnostic_never_prints_environment_values(self):
        sentinel_values = {
            "DATABASE_URL": "postgresql://private-user:private-password@example.invalid/db",
            "DEBUG": "True",
            "JWT_SECRET_KEY": "private-jwt-secret-value",
        }
        fake_dotenv = types.ModuleType("dotenv")
        fake_dotenv.load_dotenv = lambda: None
        output = io.StringIO()

        with patch.dict(os.environ, sentinel_values, clear=True), patch.dict(
            sys.modules, {"dotenv": fake_dotenv}
        ), contextlib.redirect_stdout(output):
            runpy.run_path("test_env.py", run_name="__main__")

        rendered = output.getvalue()
        for value in (sentinel_values["DATABASE_URL"], sentinel_values["JWT_SECRET_KEY"]):
            self.assertNotIn(value, rendered)
        self.assertIn("DATABASE_URL configured: True", rendered)
        self.assertIn("JWT_SECRET_KEY configured: True", rendered)


if __name__ == "__main__":
    unittest.main()
