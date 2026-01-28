# backend/config.py
import os
from functools import lru_cache
from dotenv import load_dotenv
from DBmanager1 import UserDBHandler
# Load .env explicitly (required)
load_dotenv(dotenv_path='backend/.env')

userdbhandler = UserDBHandler()
class Settings:
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.ALLOWED_USERS = os.getenv("ALLOWED_USERS")

        missing = []
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        # if not self.ALLOWED_USERS:
        #     missing.append("ALLOWED_USERS")

        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check your .env file."
            )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
