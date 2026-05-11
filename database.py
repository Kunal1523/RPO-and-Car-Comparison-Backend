# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# import os
# from dotenv import load_dotenv

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

if "sqlite" in DATABASE_URL:
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,

    # Connection health checks
    pool_pre_ping=True,
    pool_recycle=1800,

    # IMPORTANT FIXES
    pool_size=5,         # max permanent connections
    max_overflow=2,      # temporary extra connections
    pool_timeout=30,     # wait time before timeout

    # optional
    echo=False
)

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
