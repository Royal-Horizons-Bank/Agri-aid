import logging
import base64
import os
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import aiohttp
import json
import re
from io import BytesIO
from PIL import Image
import ollama  # Uses the official Ollama Python client

logger = logging.getLogger(__name__)


class ImageAnalysisService:
    """
    Analyzes agricultural images to identify pests and diseases SPECIFIC TO THE PHILIPPINES.
    Uses vision APIs (Google Vision, Claude Vision, or Ollama multimodal).
    """

    def __init__(self):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_vision_model = os.getenv("OLLAMA_VISION_MODEL", "llava")
        self.timeout = aiohttp.ClientTimeout(total=60)

        # Strictly Philippine Agricultural Pests
        self.pest_database = self._build_pest_database()
        self.disease_database = self._build_disease_database()

    def _build_pest_database(self) -> Dict:
        """Build Philippine agricultural pest database (Verified List)"""
        return {
            "armyworm": {
                "scientific_name": "Spodoptera litura / frugiperda",
                "local_name": "Harabas / Uod",
                "crops_affected": ["rice", "corn", "onion"],
                "description": "Larvae feed on leaves leaving only veins. Major pest in Nueva Ecija and Pangasinan.",
                "control_methods": ["Biological control (Trichogramma)", "Spray Bacillus thuringiensis (Bt)",
                                    "Use pheromone traps"]
            },
            "rice_black_bug": {
                "scientific_name": "Scotinophara coarctata",
                "local_name": "Itim na Atangya",
                "crops_affected": ["rice"],
                "description": "Sucks sap from the base of the plant causing 'bugburn'. Common in Bicol and Visayas.",
                "control_methods": ["Light trapping during full moon", "Herding ducks in the field",
                                    "Submerge eggs by raising water level"]
            },
            "brown_planthopper": {
                "scientific_name": "Nilaparvata lugens",
                "local_name": "Kayumangging Atangya",
                "crops_affected": ["rice"],
                "description": "Causes 'hopperburn' (browning and drying of crops). Transmits Ragged Stunt Virus.",
                "control_methods": ["Use resistant varieties (NSIC Rc)", "Avoid excessive nitrogen fertilizer",
                                    "Synchronous planting"]
            },
            "corn_borer": {
                "scientific_name": "Ostrinia furnacalis",
                "local_name": "Uod ng Mais",
                "crops_affected": ["corn"],
                "description": "Larvae bore into stalks and ears. The most destructive corn pest in PH.",
                "control_methods": ["Detasseling", "Trichogramma release", "Planting Bt Corn (if approved)"]
            },
            "cocolisap": {
                "scientific_name": "Aspidiotus rigidus",
                "local_name": "Cocolisap",
                "crops_affected": ["coconut", "lanzones"],
                "description": "Scale insects covering leaves, blocking photosynthesis. Historic outbreak in CALABARZON.",
                "control_methods": ["Pruning and burning affected parts", "Systemic trunk injection (FPA approved)",
                                    "Release of biocontrol agents"]
            },
            "mango_cecid_fly": {
                "scientific_name": "Procontarinia spp.",
                "local_name": "Kurikong",
                "crops_affected": ["mango"],
                "description": "Causes circular, brown, scab-like lesions on fruit skin.",
                "control_methods": ["Pruning overcrowded branches", "Bagging fruits early", "Proper orchard sanitation"]
            },
            "stem_borer": {
                "scientific_name": "Scirpophaga incertulas",
                "local_name": "Aksip / Stem Borer",
                "crops_affected": ["rice"],
                "description": "Larvae bore into stem causing 'deadheart' (young stage) or 'whitehead' (reproductive stage).",
                "control_methods": ["Light traps", "Pheromone traps", "Conservation of natural enemies"]
            }
        }

    def _build_disease_database(self) -> Dict:
        """Build Philippine agricultural disease database (Verified List)"""
        return {
            "rice_blast": {
                "scientific_name": "Magnaporthe oryzae",
                "local_name": "Leeg-leeg (Neck Blast)",
                "crops_affected": ["rice"],
                "description": "Diamond-shaped lesions on leaves or rotting of the panicle neck.",
                "control_methods": ["Avoid excessive nitrogen", "Keep field flooded",
                                    "Use fungicides (Tricyclazole) as last resort"]
            },
            "tungro": {
                "scientific_name": "Rice Tungro Bacilliform Virus",
                "local_name": "Tungro",
                "crops_affected": ["rice"],
                "description": "Yellow-orange discoloration of leaves, stunted growth. Vectored by Green Leafhopper.",
                "control_methods": ["Plant resistant varieties (Matatag lines)", "Control leafhopper vectors",
                                    "Roguing (removal) of infected plants"]
            },
            "bacterial_leaf_blight": {
                "scientific_name": "Xanthomonas oryzae",
                "local_name": "Kuyog",
                "crops_affected": ["rice"],
                "description": "Yellowing and drying of leaf tips and margins. Common in wet season.",
                "control_methods": ["Balanced fertilization", "Proper drainage", "Clean field sanitation"]
            },
            "panama_disease": {
                "scientific_name": "Fusarium oxysporum TR4",
                "local_name": "Fusarium Wilt",
                "crops_affected": ["banana"],
                "description": "Yellowing of older leaves, vascular discoloration. Major threat in Mindanao plantations.",
                "control_methods": ["Quarantine infected areas", "Disinfect tools/footwear",
                                    "Plant GCTCV-218 (resistant variety)"]
            }
        }

    async def analyze_image(self, image_data: bytes, filename: str, context: str = "") -> Dict:
        """
        Analyze image using Ollama Vision (LLaVA/Moondream) with ROBUST reasoning chain.
        """
        try:
            logger.info(f"Analyzing image {filename} using model: {self.ollama_vision_model}")

            # IMPROVED PROMPT: Uses Chain-of-Thought (Step-by-Step) to prevent hallucinations
            prompt = f"""
            Act as an expert Philippine Agricultural System. Analyze this image.

            Context provided: {context}

            STEP 1: VISUAL IDENTIFICATION (Reasoning)
            - Look at the image content. What is the main subject?
            - Is it a CROP (rice, corn, vegetable, fruit) or a PEST on a plant?
            - IF the image is a person, selfie, car, document, room, animal, or blurry object: MARK "is_agricultural" AS FALSE.

            STEP 2: DIAGNOSIS (Only if Agricultural)
            - Identify the specific plant.
            - Check for specific symptoms: leaf spots, yellowing, holes, or visible insects.
            - If no symptoms are visible, mark as "Healthy".

            STEP 3: GENERATE RESPONSE
            - Create a "natural_response": A helpful, polite sentence for the farmer in English.
              - If NOT agricultural, say: "I cannot analyze this. It looks like a [object], not a crop."
              - If agricultural, explain what you see naturally.

            Output VALID JSON ONLY:
            {{
                "is_agricultural": boolean,
                "plant_name": "string or 'Unknown'",
                "detected_issue": "string (e.g. 'Stem Borer', 'Rice Blast', 'None')",
                "condition": "Healthy" | "Pest Detected" | "Disease Detected" | "N/A",
                "confidence_score": number (0-100),
                "natural_response": "string"
            }}
            """

            response = ollama.chat(
                model=self.ollama_vision_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_data]
                }]
            )

            content = response['message']['content']
            logger.info(f"Ollama Raw Vision Response: {content}")

            return self._parse_llm_analysis(content)

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}")
            return self._get_fallback_analysis()

    def _parse_llm_analysis(self, content: str) -> Dict:
        """Parse the LLM's text response into structured data"""
        try:
            # Robust JSON extraction (handles cases where LLM puts text before/after JSON)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                raise ValueError("No valid JSON found in response")

            # 1. STRICT REJECTION FOR NON-AGRI
            # If explicit false, or if confidence is very low
            is_agricultural = data.get("is_agricultural", False)
            confidence = data.get("confidence_score", 100)

            if not is_agricultural or confidence < 40:
                return {
                    "plant_type": "Non-Agricultural Object",
                    "pest_detected": False,
                    "disease_detected": False,
                    "health_status": "Not Agricultural",
                    "severity": "None",
                    "recommendations": ["Please upload a clear photo of a crop, plant, or pest."],
                    "sources": [],
                    "natural_summary": data.get("natural_response",
                                                "I'm sorry, I couldn't recognize a plant or crop in this image.")
                }

            # 2. Process Agricultural Data
            plant_type = data.get("plant_name", "Unknown Crop")
            detected_issue = data.get("detected_issue", "")
            condition = data.get("condition", "Unknown")
            natural_summary = data.get("natural_response", "")

            pest_detected = "PEST" in condition.upper()
            disease_detected = "DISEASE" in condition.upper()

            # Database Matching for Recommendations
            recommendations = ["Monitor the crop closely.", "Consult your local technician."]
            pest_info = None
            disease_info = None

            content_upper = str(detected_issue).upper()

            # Match specific issue to Verified Database
            if pest_detected:
                for key, info in self.pest_database.items():
                    # Check both key and local name
                    if key.replace("_", " ").upper() in content_upper or info['local_name'].upper() in content_upper:
                        recommendations = info['control_methods']
                        pest_info = info
                        # Append specific advice to natural summary if generic
                        if len(natural_summary) < 50:
                            natural_summary += f" This resembles {info['local_name']}."
                        break

            if disease_detected:
                for key, info in self.disease_database.items():
                    if key.replace("_", " ").upper() in content_upper or info['local_name'].upper() in content_upper:
                        recommendations = info['control_methods']
                        disease_info = info
                        if len(natural_summary) < 50:
                            natural_summary += f" This resembles {info['local_name']}."
                        break

            if not pest_detected and not disease_detected:
                recommendations = ["Continue Good Agricultural Practices (GAP).", "Regular monitoring."]
                if "healthy" in condition.lower() and len(natural_summary) < 10:
                    natural_summary = f"The {plant_type} looks healthy. Keep up the good work!"

            return {
                "plant_type": plant_type,
                "pest_detected": pest_detected,
                "disease_detected": disease_detected,
                "health_status": condition,
                "severity": "Moderate" if (pest_detected or disease_detected) else "None",
                "pest_info": pest_info,
                "disease_info": disease_info,
                "recommendations": recommendations,
                "sources": [],
                "natural_summary": natural_summary
            }

        except Exception as e:
            logger.error(f"JSON Parse error: {str(e)}")
            # Fallback to text parsing if JSON fails but might still be valid text
            return self._get_fallback_analysis()

    def _get_fallback_analysis(self) -> Dict:
        """Strict fallback that admits failure"""
        return {
            "plant_type": "Analysis Failed",
            "pest_detected": False,
            "disease_detected": False,
            "health_status": "System Error",
            "natural_summary": "I'm sorry, I couldn't analyze that image clearly. Please try again with a better photo.",
            "recommendations": [],
            "sources": []
        }


# Initialize service
image_analysis_service = ImageAnalysisService()