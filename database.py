# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import os
# from dotenv import load_dotenv
 
# from config import get_settings
 
# settings = get_settings()
# DATABASE_URL = settings.DATABASE_URL
 

# from config import get_settings

# settings = get_settings()
# DATABASE_URL = settings.DATABASE_URL

# # Supabase Connect Args
# connect_args = {}
# if "sqlite" in DATABASE_URL:
#     connect_args = {"check_same_thread": False}
 

# engine = create_engine(
#     DATABASE_URL, connect_args=connect_args,pool_pre_ping=True,
#     pool_recycle=1800,
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
 
# Base = declarative_base()
 

# Base = declarative_base()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import get_settings
 
settings = get_settings()
DATABASE_URL = settings.DATABASE_URL
 
# Supabase Connect Args
connect_args = {}

_DB_AVAILABLE = bool(DATABASE_URL)

if not _DB_AVAILABLE:
    print("[database] WARNING: DATABASE_URL not set — using in-memory SQLite fallback. All DB routes will be unavailable.")
    DATABASE_URL = "sqlite:///:memory:"
    connect_args = {"check_same_thread": False}
elif "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=2,
        max_overflow=0,
        pool_timeout=30,
        echo=False
    )
except Exception as e:
    print(f"[database] WARNING: Engine creation failed ({e}) — falling back to in-memory SQLite.")
    _DB_AVAILABLE = False
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()