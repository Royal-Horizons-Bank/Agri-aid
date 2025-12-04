# backend/tests/test_chat.py

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["app"] == "Agri-Aid"

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ollama_url" in data
        assert "model" in data


class TestChatEndpoint:
    """Test chat endpoint"""

    def test_chat_valid_request(self):
        """Test chat endpoint with valid request"""
        payload = {
            "message": "Kumusta ang panahon sa Pangasinan?",
            "location": "Tayug, Pangasinan"
        }
        response = client.post("/api/v1/chat", json=payload)
        # Status might be 503 if Ollama is not running, that's OK for testing
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "conversation_id" in data
            assert "timestamp" in data
            assert data["model"] == "llama3.1:8b"

    def test_chat_invalid_request(self):
        """Test chat endpoint with invalid request (empty message)"""
        payload = {
            "message": ""
        }
        response = client.post("/api/v1/chat", json=payload)
        # Should return validation error
        assert response.status_code == 422

    def test_chat_missing_message(self):
        """Test chat endpoint with missing message field"""
        payload = {
            "location": "Manila"
        }
        response = client.post("/api/v1/chat", json=payload)
        # Should return validation error
        assert response.status_code == 422

    def test_chat_with_conversation_id(self):
        """Test chat with existing conversation ID"""
        payload = {
            "message": "Magkano ang bigas ngayon?",
            "conversation_id": "test_conv_123"
        }
        response = client.post("/api/v1/chat", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert data["conversation_id"] == "test_conv_123"


class TestStatusEndpoint:
    """Test status endpoint"""

    def test_get_status(self):
        """Test get status endpoint"""
        response = client.get("/api/v1/status")
        # Should return either 200 or 503 depending on Ollama availability
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data


class TestBatchChat:
    """Test batch chat endpoint"""

    def test_batch_chat_valid(self):
        """Test batch chat with valid requests"""
        payload = [
            {"message": "Ano ang presyo ng palay?", "location": "Cabanatuan"},
            {"message": "Kailan dapat magtanim?"}
        ]
        response = client.post("/api/v1/chat/batch", json=payload)
        # Status might vary based on Ollama
        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "results" in data
            assert data["total"] == 2


@pytest.mark.asyncio
async def test_ollama_service_initialization():
    """Test Ollama service can be initialized"""
    from app.services.ollama_service import OllamaService
    try:
        service = OllamaService()
        status = await service.get_ollama_status()
        assert "status" in status
    except Exception as e:
        # OK if Ollama is not running
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
