from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import configuration
from app.config import settings

# Import routers
from app.routers import chat, image_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events: Startup and Shutdown logic.
    Replaces deprecated @app.on_event("startup").
    """
    # Startup
    logger.info(f"🌾 {settings.APP_NAME} API starting up...")
    logger.info(f"🔧 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🤖 Vision Model: {settings.OLLAMA_VISION_MODEL}")
    logger.info("✅ Services initialized")

    yield

    # Shutdown
    logger.info(f"💤 {settings.APP_NAME} API shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version="3.2",
    description="Agricultural AI Assistant Backend with Vision & Real-time Data",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(image_router.router)


@app.get("/")
async def root():
    """Root endpoint to verify service status"""
    return {
        "message": f"Welcome to {settings.APP_NAME} API",
        "status": "active",
        "version": "3.2",
        "docs": "/docs",
        "services": {
            "chat": "active",
            "image_analysis": "active",
            "weather": "active (PAGASA/Philippines Only)",
            "market_prices": "active (DA/PSA Fallback)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agri-aid",
        "timestamp": "now"
    }