import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from app.logger import setup_logger

logger = setup_logger(__name__)

# Base model
Base = declarative_base()

class JobDescription(Base):
    __tablename__ = 'job_descriptions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ResumeAnalysis(Base):
    __tablename__ = 'resume_analyses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey('job_descriptions.id', ondelete='CASCADE'), nullable=True)
    filename = Column(String(255), nullable=False)
    candidate_name = Column(String(255), nullable=True)
    
    # Core scores
    final_match_score = Column(Float, default=0.0)
    ai_plausibility_score = Column(Float, default=0.5)
    classification = Column(String(50), nullable=False)
    
    # JSON payload of all features/results for the frontend
    full_results = Column(JSON, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Async Engine and Session
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'resume_screener.db')
# Ensure data dir exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized successfully.")

async def get_db():
    async with async_session() as session:
        yield session
