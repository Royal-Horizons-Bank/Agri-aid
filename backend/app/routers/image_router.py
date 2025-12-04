from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
from datetime import datetime
from app.services.image_analysis_service import image_analysis_service
from app.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["image"])

# Initialize Ollama service
try:
    ollama_service = OllamaService()
except Exception as e:
    logger.error(f"Failed to initialize OllamaService: {str(e)}")
    ollama_service = None


# ==================== REQUEST/RESPONSE MODELS ====================

class ImageAnalysisResponse(BaseModel):
    """Response for image analysis"""
    plant_type: str = Field(..., description="Identified plant type")
    pest_detected: bool = Field(..., description="Whether pests were detected")
    pest_info: Optional[Dict[str, Any]] = Field(None, description="Pest information if detected")
    disease_detected: bool = Field(..., description="Whether diseases were detected")
    disease_info: Optional[Dict[str, Any]] = Field(None, description="Disease information if detected")
    health_status: str = Field(..., description="Overall plant health status")
    severity: Optional[str] = Field(None, description="Severity level")
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Source references")
    natural_summary: Optional[str] = Field(None, description="Natural language summary of the analysis")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DetailedAnalysisResponse(BaseModel):
    """Detailed analysis with follow-up information"""
    initial_analysis: ImageAnalysisResponse
    detailed_info: str = Field(..., description="Detailed information about identified pest/disease")
    control_methods: List[str] = Field(..., description="Control/treatment methods")
    preventive_measures: List[str] = Field(..., description="Preventive measures")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="Source references")


# ==================== ENDPOINTS ====================

@router.post("/analyze-image", response_model=ImageAnalysisResponse, summary="Analyze agricultural image")
async def analyze_image(
        file: UploadFile = File(..., description="Image file (JPG, PNG)"),
        location: Optional[str] = Form(None, description="Farm location"),
        context: Optional[str] = Form(None, description="Additional context")
):
    """
    Analyze agricultural image using robust Vision AI.
    """
    try:
        # Validate file type
        allowed_formats = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
        if file.content_type not in allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: JPG, PNG, WEBP"
            )

        # Read file bytes directly
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 10MB limit"
            )

        logger.info(f"Analyzing image: {file.filename} from {location}")

        # Prepare Context
        analysis_context = context or ""
        if location:
            analysis_context += f" Location: {location}"

        # Call Service (Pass bytes directly)
        result = await image_analysis_service.analyze_image(
            image_data=contents,
            filename=file.filename,
            context=analysis_context
        )

        if not result:
            raise HTTPException(status_code=500, detail="Analysis failed to return results")

        # Handle "Not Agricultural" case gracefully without 500 error
        if result.get("health_status") == "Not Agricultural":
            return ImageAnalysisResponse(
                plant_type="N/A",
                pest_detected=False,
                disease_detected=False,
                health_status="Not Agricultural",
                severity="None",
                recommendations=["Please upload a valid image of a crop or plant."],
                sources=[],
                natural_summary=result.get("natural_summary", "This image does not appear to be agricultural.")
            )

        # Default Sources if service returns empty (Fallback to trusted PH agencies)
        sources = result.get("sources", [])
        if not sources:
            sources = [
                {"title": "PhilRice - Rice Knowledge Bank", "url": "https://www.philrice.gov.ph"},
                {"title": "Bureau of Plant Industry (BPI)", "url": "https://bpi.gov.ph"}
            ]

        # Format response
        return ImageAnalysisResponse(
            plant_type=result.get("plant_type", "Unknown"),
            pest_detected=result.get("pest_detected", False),
            pest_info=result.get("pest_info"),
            disease_detected=result.get("disease_detected", False),
            disease_info=result.get("disease_info"),
            health_status=result.get("health_status", "Unknown"),
            severity=result.get("severity", "None"),
            recommendations=result.get("recommendations", []),
            sources=sources,
            natural_summary=result.get("natural_summary", result.get("health_status"))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


@router.post("/detailed-pest-info", response_model=DetailedAnalysisResponse)
async def get_detailed_info(
        pest_name: str = Form(..., description="Pest or disease name"),
        plant_type: str = Form(..., description="Plant type"),
        location: Optional[str] = Form(None, description="Farm location")
):
    """
    Get detailed info using text-based LLM for a detected pest.
    """
    try:
        if not ollama_service:
            raise HTTPException(status_code=503, detail="Ollama service unavailable")

        # Prompt for detailed, natural explanation
        detailed_prompt = f"""
        Provide a detailed, practical guide for farmers about this issue:
        Pest/Disease: {pest_name}
        Plant: {plant_type}
        Location Context: {location or 'Philippines'}

        Structure the response:
        1. **Identification**: Key visual signs.
        2. **Damage**: How it affects the crop yield.
        3. **Management**: Concrete steps (Cultural, Biological, Chemical as last resort).
        4. **Prevention**: How to stop it from coming back.

        Keep the tone professional yet accessible to farmers.
        """

        response_obj = await ollama_service.generate_response(detailed_prompt, location)

        # Construct a placeholder initial analysis for the response model structure
        initial_analysis = ImageAnalysisResponse(
            plant_type=plant_type,
            pest_detected=True,
            disease_detected=False,
            health_status="Detailed Inquiry",
            recommendations=[],
            sources=[]
        )

        return DetailedAnalysisResponse(
            initial_analysis=initial_analysis,
            detailed_info=response_obj.text,
            control_methods=["See detailed guide above"],
            preventive_measures=["Regular monitoring", "Field sanitation"],
            sources=[
                {"title": "PhilRice", "url": "https://www.philrice.gov.ph"},
                {"title": "Bureau of Plant Industry", "url": "https://bpi.gov.ph"}
            ]
        )

    except Exception as e:
        logger.error(f"Detailed info error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))