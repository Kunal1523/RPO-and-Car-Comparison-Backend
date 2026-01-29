from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

from config import get_settings

settings = get_settings()
DATABASE_URL = settings.DATABASE_URL

# Supabase Connect Args
connect_args = {}
if "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL, connect_args=connect_args,pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
