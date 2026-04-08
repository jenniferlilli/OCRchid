from sqlalchemy import create_engine, Column, Integer, Boolean, String, ForeignKey, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
import os
from sqlalchemy.dialects.postgresql import UUID
import uuid

load_dotenv()

Base = declarative_base()

class ValidBadgeIDs(Base):
    __tablename__ = 'valid_badge_ids'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'), nullable=False)
    badge_id = Column(String)

class Ballot(Base):
    __tablename__ = 'ballots'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'), nullable=False)
    badge_id = Column(String)
    badge_status = Column(String)
    s3_key = Column(String)
    name = Column(String)
    validity = Column(Boolean)
    created_at = Column(DateTime, server_default=func.current_timestamp())

class UploadedZip(Base):
    __tablename__ = 'uploaded_zips'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'), nullable=False)
    filename = Column(String)

class UserSession(Base):
    __tablename__ = 'sessions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), unique=True, nullable=False, default=uuid.uuid4)
    password = Column(String, nullable=False)
    spreadsheet_id = Column(String, nullable=True)

class OCRResult(Base):
    __tablename__ = 'ocr_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.session_id'), nullable=False)
    filename = Column(String)
    extracted_text = Column(String)

class BallotVotes(Base):
    __tablename__ = 'votes'
    id = Column(Integer, primary_key=True, autoincrement=True)
    badge_id = Column(String)
    ballot_id = Column(Integer, ForeignKey('ballots.id'), nullable=False)
    name = Column(String)
    category_id = Column(String)
    vote = Column(String)
    key = Column(String)
    vote_status = Column(String)
    is_valid = Column(Boolean)

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True, autoincrement=True)
    category_id = Column(String, nullable=False)
    product_number = Column(String, nullable=False)
    product_name = Column(String, nullable=False)


DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not defined in .env file")

print(f"Connecting to database at: {DATABASE_URL}")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

Base.metadata.create_all(engine)

SessionLocal = scoped_session(sessionmaker(bind=engine))

def get_db_session():
    return SessionLocal()
