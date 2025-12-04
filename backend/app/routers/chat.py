from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict
import logging
from datetime import datetime
from app.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    location: Optional[str] = Field(None, max_length=100, description="User location (province)")
    language: str = Field(default="en", description="Response language")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class SourceReference(BaseModel):
    """Individual source reference with clickable link"""
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Clickable URL to source")
    snippet: Optional[str] = Field(None, description="Brief excerpt from source")
    source_type: str = Field(default="web", description="Type: official, web, news, research")


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI-generated response")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str = Field(default="llama3.1:8b", description="LLM model used")
    location: Optional[str] = Field(None, description="Location context")
    sources: List[Union[SourceReference, str]] = Field(default_factory=list, description="List of sources used")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "The current buying price of palay in Nueva Ecija is ₱23.00/kg.",
                "conversation_id": "conv_12345",
                "timestamp": "2024-11-14T10:00:00",
                "model": "llama3.1:8b",
                "location": "Nueva Ecija",
                "sources": [
                    "https://www.da.gov.ph/price-monitoring/",
                    "https://openstat.psa.gov.ph"
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
    code: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ==================== ENDPOINTS ====================

@router.post("/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat_endpoint(request: ChatRequest):
    """
    Process chat message using local Ollama LLM with Philippine-specific data integration.
    """
    try:
        logger.info(f"Chat request: {request.message[:50]}... | Location: {request.location}")

        if not ollama_service:
            raise HTTPException(status_code=503, detail="AI Service unavailable")

        # Generate response using the enhanced service method
        # This method includes the logic for:
        # 1. Strict Philippine-only context
        # 2. Price monitoring (DA/PSA)
        # 3. Weather (PAGASA via Serper)
        # 4. Web Search fallback
        response_data = await ollama_service.generate_response_with_data(
            message=request.message,
            location=request.location
        )

        # Format sources for response
        # The service might return strings or objects, we handle both
        formatted_sources = []
        if response_data.sources:
            for src in response_data.sources:
                if isinstance(src, str):
                    # Convert string URL to SourceReference object if possible
                    # or just pass the string (ChatResponse model supports both)
                    formatted_sources.append(src)
                elif isinstance(src, dict):
                    formatted_sources.append(SourceReference(**src))
                else:
                    # Handle SourceLink objects from search_service
                    try:
                        formatted_sources.append(src.to_dict())
                    except AttributeError:
                        formatted_sources.append(str(src))

        logger.info(f"Response generated with {len(formatted_sources)} sources")

        return ChatResponse(
            response=response_data.text,
            conversation_id=request.conversation_id or f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            location=request.location,
            sources=formatted_sources
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "service": "agri-aid-chat",
        "timestamp": datetime.utcnow()
    }

    # Optional: Check Ollama connection
    if ollama_service:
        ollama_status = await ollama_service.get_ollama_status()
        status["ollama"] = ollama_status
    else:
        status["ollama"] = {"status": "not_initialized"}

    return status