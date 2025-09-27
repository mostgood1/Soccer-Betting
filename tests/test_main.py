import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db, Base

# Create in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Soccer Betting Platform API", "version": "1.0.0"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_get_matches():
    response = client.get("/api/v1/matches/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_teams():
    response = client.get("/api/v1/teams/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_players():
    response = client.get("/api/v1/players/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_leagues():
    response = client.get("/api/v1/leagues/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_predictions():
    response = client.get("/api/v1/predictions/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)