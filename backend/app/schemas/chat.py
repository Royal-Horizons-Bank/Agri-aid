from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    location: Optional[str] = Field(None, description="Agricultural location/region")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Ano ang presyo ng palay ngayong taon?",
                "location": "Tayug, Pangasinan",
                "conversation_id": None
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI-generated response")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str = Field(default="llama3.1:8b", description="LLM model used")
    location: Optional[str] = Field(None, description="Location context")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Ang presyo ng palay ay umabot na sa ₱20-25 per kilo depende sa kalidad...",
                "conversation_id": "conv_abc123def456",
                "timestamp": "2025-11-13T15:30:00",
                "model": "llama3.1:8b",
                "location": "Tayug, Pangasinan"
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Failed to connect to Ollama server",
                "code": "OLLAMA_CONNECTION_ERROR",
                "timestamp": "2025-11-13T15:30:00"
            }
        }
