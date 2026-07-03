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
        self.GUEST_EMAIL_1 = os.getenv("GUEST_EMAIL_1")
        self.GUEST_EMAIL_2 = os.getenv("GUEST_EMAIL_2")

        if not self.DATABASE_URL:
            print("[config] WARNING: DATABASE_URL is not set — database features will be unavailable.")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
