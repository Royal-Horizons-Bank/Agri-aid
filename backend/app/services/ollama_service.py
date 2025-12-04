from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
import logging
from typing import Optional, Dict, List, Tuple
from app.config import settings
from app.services.price_scraper import psa_scraper
from app.services.weather_service import weather_service
from app.services.search_service import search_service
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# ==================== COMPREHENSIVE PHILIPPINE AGRICULTURE OFFICIALS DATABASE ====================

AGRICULTURE_OFFICIALS = {
    "National": {
        "DA": {
            "name": "Department of Agriculture",
            "secretary": "Secretary Francisco Tiu Laurel Jr.",
            "position": "Secretary of Agriculture",
            "website": "https://www.da.gov.ph/about-us/directory-of-officials/",
            "hotline": "1888",
            "email": "info@da.gov.ph",
            "address": "Diliman, Quezon City",
            "phone": "(02) 8928-1234"
        },
        "PhilRice": {
            "name": "Philippine Rice Research Institute",
            "director": "Dr. John de Leon",
            "position": "Executive Director",
            "website": "https://www.philrice.gov.ph/about-us/directory/",
            "address": "Science City of Muñoz, Nueva Ecija",
            "phone": "(044) 456-0277",
            "email": "prri.mail@philrice.gov.ph"
        },
        "ATI": {
            "name": "Agricultural Training Institute",
            "director": "Remelyn R. Recoter",
            "position": "Director IV",
            "website": "https://ati2.da.gov.ph/ati-main/content/directory-key-officials",
            "address": "Diliman, Quezon City",
            "phone": "(02) 8929-8541"
        },
        "BPI": {
            "name": "Bureau of Plant Industry",
            "director": "Gerald Glenn F. Panganiban",
            "position": "Director",
            "website": "https://bpi.gov.ph/index.php/about-us/officials",
            "address": "Malate, Manila",
            "phone": "(02) 8525-7313"
        },
        "BAI": {
            "name": "Bureau of Animal Industry",
            "director": "Dr. Paul C. Limson",
            "position": "Director",
            "website": "https://bai.gov.ph/index.php/contact-us",
            "address": "Quezon City",
            "phone": "(02) 8928-2429"
        },
        "PSA": {
            "name": "Philippine Statistics Authority",
            "administrator": "Claire Dennis S. Mapa",
            "position": "National Statistician",
            "website": "https://psa.gov.ph/statistics/crops/palay-and-corn-estimates",
            "phone": "(02) 8462-6600",
            "email": "info@psa.gov.ph"
        }
    },
    # ... (Regional and Provincial data retained as is)
    "Regional": {
        "Region I": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region I",
            "location": "Dagupan City, Pangasinan",
            "regions_covered": ["Ilocos Norte", "Ilocos Sur", "La Union", "Pangasinan"],
            "contact": "(075) 516-2456",
            "website": "https://ilocos.da.gov.ph/contact-us/"
        },
        "Region II": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region II",
            "location": "Santiago City, Isabela",
            "regions_covered": ["Isabela", "Nueva Vizcaya", "Quirino", "Batanes"],
            "contact": "(078) 608-3434",
            "website": "https://cagayan.da.gov.ph/contact-us/"
        },
        "Region III": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region III",
            "location": "Cabanatuan City, Nueva Ecija",
            "regions_covered": ["Nueva Ecija", "Bulacan", "Pampanga", "Batangas", "Laguna", "Cavite", "Rizal"],
            "contact": "(044) 463-5100",
            "website": "https://centralluzon.da.gov.ph/directory/"
        },
        "Region IV-A": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region IV-A",
            "location": "Calamba, Laguna",
            "regions_covered": ["Cavite", "Laguna", "Batangas", "Rizal", "Quezon"],
            "contact": "(049) 501-0556",
            "website": "https://calabarzon.da.gov.ph/contact-us/"
        },
        "Region V": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region V",
            "location": "Legazpi City, Albay",
            "regions_covered": ["Albay", "Camarines Norte", "Camarines Sur", "Sorsogon"],
            "contact": "(052) 481-4269",
            "website": "https://bicol.da.gov.ph/directory/"
        },
        "Region VI": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region VI",
            "location": "Iloilo City, Iloilo",
            "regions_covered": ["Iloilo", "Capiz", "Antique", "Aklan", "Guimaras"],
            "contact": "(033) 335-1208",
            "website": "https://westernvisayas.da.gov.ph/directory/"
        },
        "Region VII": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region VII",
            "location": "Cebu City, Cebu",
            "regions_covered": ["Cebu", "Bohol", "Negros Oriental", "Siquijor"],
            "contact": "(032) 254-0910",
            "website": "https://centralvisayas.da.gov.ph/directory/"
        },
        "Region VIII": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region VIII",
            "location": "Tacloban City, Leyte",
            "regions_covered": ["Leyte", "Southern Leyte", "Samar", "Northern Samar", "Eastern Samar"],
            "contact": "(053) 321-2567",
            "website": "https://easternvisayas.da.gov.ph/directory/"
        },
        "Region IX": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region IX",
            "location": "Zamboanga City, Zamboanga del Sur",
            "regions_covered": ["Zamboanga del Sur", "Zamboanga del Norte", "Zamboanga Sibugay"],
            "contact": "(062) 992-0213",
            "website": "https://zamboanga.da.gov.ph/directory/"
        },
        "Region X": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region X",
            "location": "Cagayan de Oro City, Misamis Oriental",
            "regions_covered": ["Misamis Oriental", "Bukidnon", "Misamis Occidental", "Lanao del Norte"],
            "contact": "(088) 858-6157",
            "website": "https://northernmindanao.da.gov.ph/directory/"
        },
        "Region XI": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region XI",
            "location": "Davao City, Davao del Sur",
            "regions_covered": ["Davao del Sur", "Davao Oriental", "Davao del Norte"],
            "contact": "(082) 304-4242",
            "website": "https://davao.da.gov.ph/directory/"
        },
        "Region XII": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region XII",
            "location": "Koronadal City, South Cotabato",
            "regions_covered": ["South Cotabato", "Sarangani", "Sultan Kudarat", "Cotabato"],
            "contact": "(083) 228-0932",
            "website": "https://soccsksargen.da.gov.ph/directory/"
        },
        "Region XIII": {
            "head": "OIC/Provincial Director",
            "office": "Regional Field Office - Region XIII",
            "location": "Butuan City, Agusan del Norte",
            "regions_covered": ["Agusan del Norte", "Agusan del Sur", "Surigao del Norte", "Surigao del Sur"],
            "contact": "(085) 342-7532",
            "website": "https://caraga.da.gov.ph/directory/"
        }
    },

    "Provincial": {
        "Pangasinan": {
            "pao_head": "Ms. Dalisay A. Moya",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Pangasinan Capitol Compound, San Carlos, Pangasinan",
            "website": "https://www.pangasinan.gov.ph/department/provincial-agriculture-office/",
            "phone": "(075) 606-0850",
            "region": "Region I"
        },
        "Nueva Ecija": {
            "pao_head": "Provincial Agriculture Officer",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Nueva Ecija Capitol Compound, Cabanatuan City",
            "region": "Region III",
            "phone": "(044) 463-0849"
        },
        "Isabela": {
            "pao_head": "Provincial Agriculture Officer",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Isabela Capitol Compound, Ilagan, Isabela",
            "region": "Region II",
            "phone": "(078) 622-1234"
        },
        "Laguna": {
            "pao_head": "Provincial Agriculture Officer",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Laguna Capitol Compound, Santa Cruz, Laguna",
            "region": "Region IV-A",
            "phone": "(049) 501-7542"
        },
        "Iloilo": {
            "pao_head": "Provincial Agriculture Officer",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Iloilo Capitol Compound, Iloilo City",
            "region": "Region VI",
            "phone": "(033) 335-1208"
        },
        "Cebu": {
            "pao_head": "Provincial Agriculture Officer",
            "position": "Provincial Agriculture Officer",
            "office": "Provincial Agriculture Office",
            "address": "Cebu Provincial Capitol, Cebu City",
            "region": "Region VII",
            "phone": "(032) 254-0910"
        }
    }
}

# ==================== COMPREHENSIVE CROP DATABASE ====================
# (Updated with deeper links where applicable)

CROP_DATABASE = {
    "palay": {
        "common_names": ["rice", "palay", "bigas", "padi"],
        "tagalog": "Palay",
        "season_wet": "June-November",
        "season_dry": "December-May",
        "ideal_temperature": "23-32°C",
        "water_requirement": "1000-1500mm annually",
        "soil_type": "Silty loam to clay loam",
        "ph_range": "5.5-7.0",
        "major_producing_regions": ["Nueva Ecija", "Pangasinan", "Isabela", "Cagayan", "Laguna"],
        "average_yield": "3.5-5 tons per hectare",
        "pests": ["armyworms", "rice blast", "tungro virus", "brown planthopper", "leaf folder", "stem borer"],
        "diseases": ["bacterial leaf blight", "rice blast", "rice tungro virus", "false smut", "leaf scald"],
        "best_practices": [
            "Use certified quality seeds (BAS/RBS)",
            "Proper land preparation and puddling",
            "Optimal water management (25-50mm standing water)",
            "Balanced NPK fertilizer application (60-90-60 kg/ha)",
            "Timely pest and disease control",
            "Proper harvesting at 85-90% grain maturity",
            "Correct drying (down to 14% moisture)",
            "Proper storage in cool, dry place"
        ],
        "varieties": ["IR64", "IR66", "PSB Rc50", "PSB Rc40", "PSB Rc32", "Sinandomeng"],
        "harvesting_period": "110-120 days",
        "source": "https://www.philrice.gov.ph/ricelytics/"
    },
    "corn": {
        "common_names": ["corn", "maize", "mais", "corn on the cob"],
        "tagalog": "Mais",
        "season": "March-August (dry season)",
        "ideal_temperature": "21-32°C",
        "water_requirement": "600-800mm",
        "soil_type": "Well-drained, fertile soil",
        "ph_range": "6.0-7.5",
        "major_producing_regions": ["Cagayan", "Nueva Ecija", "Laguna", "Isabela", "Bukidnon"],
        "average_yield": "3-5 tons per hectare",
        "pests": ["corn borer", "cutworm", "armyworm", "grasshopper"],
        "diseases": ["corn leaf rust", "leaf spot", "maize dwarf mosaic virus"],
        "varieties": ["Pioneer", "Dekalb", "Hybrid DMH", "Hybrid PSC"],
        "harvesting_period": "80-120 days",
        "best_practices": [
            "Use improved hybrid seeds",
            "Proper spacing (75cm x 20cm)",
            "Balanced fertilizer (14-14-14)",
            "Weed management",
            "Irrigation during critical periods",
            "Harvest when kernels are hard and silk turns brown"
        ],
        "source": "https://www.da.gov.ph/program/national-corn-program/"
    },
    "coconut": {
        "common_names": ["coconut", "niyog"],
        "tagalog": "Niyog",
        "ideal_temperature": "24-32°C",
        "water_requirement": "1500-2250mm",
        "major_producing_regions": ["Quezon", "Laguna", "Cavite", "Batangas", "Catanduanes"],
        "average_yield": "6000-12000 nuts per hectare per year",
        "products": ["coconut oil", "copra", "coconut milk", "coir", "coconut water", "activated charcoal"],
        "harvesting_period": "6-7 years for first harvest, yearly thereafter",
        "average_lifespan": "60-80 years",
        "source": "https://pca.gov.ph/index.php/resources/coconut-statistics"
    },
    "sugarcane": {
        "common_names": ["sugarcane", "tubo"],
        "tagalog": "Tubo",
        "harvest_season": "October-March",
        "ideal_temperature": "20-30°C",
        "water_requirement": "1500-2250mm",
        "major_producing_regions": ["Negros", "Isabela", "Bukidnon", "Quezon"],
        "average_yield": "50-60 tons per hectare",
        "harvesting_period": "12-18 months",
        "varieties": ["Iloilo", "Victoriano", "PH02-3250"],
        "uses": ["sugar production", "molasses", "ethanol", "animal feed"],
        "source": "https://www.sra.gov.ph/industry-update/"
    }
}

# ==================== GOVERNMENT PROGRAMS DATABASE ====================

GOVERNMENT_PROGRAMS = {
    "DA_Programs": [
        {
            "name": "Rice Competitiveness Enhancement Fund (RCEF)",
            "description": "Support for rice farmers through mechanization and inputs",
            "beneficiaries": "Rice farmers",
            "budget": "Billions annually",
            "source": "https://rcef.da.gov.ph/"
        },
        {
            "name": "High Value Crops Development Program",
            "description": "Support for fruits, vegetables, and spices production",
            "beneficiaries": "High-value crop farmers",
            "link": "https://hvcdp.da.gov.ph/",
            "source": "https://hvcdp.da.gov.ph/"
        },
        {
            "name": "Livestock and Poultry Program",
            "description": "Support for livestock and poultry production",
            "beneficiaries": "Livestock farmers",
            "source": "https://nlp.da.gov.ph/"
        },
        {
            "name": "Agricultural Mechanization Service",
            "description": "Equipment support and training",
            "beneficiaries": "All farmers",
            "source": "https://philmech.gov.ph/"
        },
        {
            "name": "Smallholder Irrigation Support Program (SISP)",
            "description": "Irrigation infrastructure development",
            "beneficiaries": "Small farmers",
            "source": "https://bswm.da.gov.ph/"
        },
        {
            "name": "Climate Change Adaptation and Mitigation Program",
            "description": "Climate-resilient agriculture practices",
            "beneficiaries": "All farmers",
            "source": "https://climateresilient.da.gov.ph/"
        },
        {
            "name": "Agricultural Credit Policy Council (ACPC)",
            "description": "Agricultural financing and credit assistance",
            "beneficiaries": "All farmers",
            "source": "https://acpc.gov.ph/"
        }
    ],
    "financing_options": [
        {
            "name": "LandBank of the Philippines",
            "type": "Agricultural Loans",
            "website": "https://www.landbank.com/loans/agricultural-lending",
            "phone": "1-800-LANDBANK"
        },
        {
            "name": "DBP - Development Bank of the Philippines",
            "type": "Agricultural Development Loans",
            "website": "https://www.dbp.ph/developmental-banking/agricultural-sector/",
            "phone": "(02) 8708-7000"
        },
        {
            "name": "QUEDANCOR - Quedan and Rural Credit Guarantee Corporation",
            "type": "Credit Guarantee and Financing",
            "website": "https://quedancor.gov.ph/",
            "phone": "(02) 8928-1111"
        },
        {
            "name": "COA - Cooperative Development Authority",
            "type": "Cooperative Financing",
            "website": "https://cda.gov.ph/",
            "phone": "(02) 8942-6210"
        }
    ]
}

# ==================== PEST & DISEASE MANAGEMENT ====================

PEST_DISEASE_DATABASE = {
    "major_pests": {
        "armyworm": {
            "crops": ["palay", "corn", "sugarcane"],
            "damage": "Leaf feeding, stem tunneling, severe defoliation",
            "identification": "Greenish-brown caterpillar, 30-35mm long",
            "control": [
                "Biological control using parasitic wasps",
                "Insecticidal spray (Bacillus thuringiensis)",
                "Proper field sanitation",
                "Early removal of affected plants"
            ],
            "source": "https://bpi.gov.ph/"
        },
        "brown_planthopper": {
            "crops": ["palay"],
            "damage": "Sap feeding, virus transmission, hopper burn",
            "identification": "Small brown insects on leaf undersides",
            "control": [
                "Plant resistant varieties",
                "Insecticide application (neem oil)",
                "Water management (avoid excessive water)",
                "Light traps"
            ],
            "source": "https://www.philrice.gov.ph/ricelytics/"
        },
        "rice_blast": {
            "crops": ["palay"],
            "type": "fungal disease",
            "damage": "Gray diamond-shaped lesions on leaves and panicles",
            "control": [
                "Use resistant varieties",
                "Fungicide spray (Tricyclazole, Azoxystrobin)",
                "Field sanitation",
                "Avoid excessive nitrogen"
            ],
            "source": "https://www.philrice.gov.ph/ricelytics/"
        },
        "corn_borer": {
            "crops": ["corn"],
            "damage": "Tunnel boring in stems and ears",
            "control": [
                "Early planting",
                "Proper field sanitation",
                "Insecticidal spray",
                "Biological control"
            ],
            "source": "https://bpi.gov.ph/"
        }
    },
    "disease_management": {
        "integrated_pest_management": [
            "Cultural practices (crop rotation, field sanitation)",
            "Biological control (beneficial insects)",
            "Chemical control (pesticides as last resort)",
            "Use of resistant varieties",
            "Regular monitoring and scouting"
        ],
        "organic_alternatives": [
            "Neem oil spray",
            "Bacillus thuringiensis (Bt)",
            "Entomopathogenic fungi",
            "Trap cropping",
            "Hand-picking",
            "Companion planting"
        ]
    }
}

# ==================== SEASONAL GUIDE ====================

SEASONAL_GUIDE = {
    "wet_season": {
        "months": "June-November",
        "temperature": "25-29°C",
        "rainfall": "1500-2500mm",
        "crops": ["Palay (rice)", "corn", "vegetables", "upland crops"],
        "planting_activities": [
            "Prepare seedbeds in late May",
            "Land preparation starts June",
            "Transplanting in late June-July",
            "Weeding and maintenance July-September",
            "Fertilizer application"
        ],
        "pest_concerns": ["Leaf folder", "Brown planthopper", "Rice blast", "Bacterial leaf blight"],
        "harvesting": ["September-October for early plantings", "October-November for later plantings"],
        "precautions": [
            "Monitor water levels (25-50mm standing water)",
            "Disease prevention",
            "Regular pest monitoring",
            "Proper drainage in low areas"
        ],
        "source": "https://www.da.gov.ph/"
    },
    "dry_season": {
        "months": "December-May",
        "temperature": "22-35°C",
        "rainfall": "100-200mm (minimal)",
        "crops": ["Corn (mais)", "legumes", "vegetables", "root crops"],
        "planting_activities": [
            "Land preparation December-January",
            "Planting January-February",
            "Irrigation planning and installation",
            "Fertilizer and soil amendment"
        ],
        "water_management": [
            "Plan irrigation schedule",
            "Check water availability",
            "Maintain irrigation systems",
            "Mulching for water retention"
        ],
        "precautions": [
            "Water management (critical)",
            "Soil conservation",
            "Dust control",
            "Monitor for drought stress"
        ],
        "source": "https://www.da.gov.ph/"
    }
}


class ResponseWithSources:
    """Response object with source attribution"""

    def __init__(self, text: str, sources: List[str] = None):
        self.text = text
        self.sources = sources or []

    def to_dict(self) -> Dict:
        return {
            "response": self.text,
            "sources": self.sources
        }


class OllamaService:
    """Comprehensive Ollama service with source attribution and hallucination prevention"""

    def __init__(self):
        """Initialize Ollama service"""
        try:
            self.llm = Ollama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL,
                temperature=settings.OLLAMA_TEMPERATURE
            )
            self.memory = ConversationBufferWindowMemory(
                k=10,
                return_messages=True,
                memory_key="chat_history"
            )
            logger.info(f"Ollama service initialized with model: {settings.OLLAMA_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama service: {str(e)}")
            raise

    def _track_sources(self, context: str) -> List[str]:
        """Extract sources from context"""
        sources = []
        urls = re.findall(r'https?://[^\s\)]+', context)
        sources.extend(urls)
        return list(set(sources))

    def _get_knowledge_base_info(self, query: str, location: Optional[str] = None) -> Tuple[str, List[str]]:
        """Extract comprehensive agricultural information from knowledge base"""
        knowledge_context = ""
        sources = []
        query_lower = query.lower()

        # Check for official/personnel queries
        if any(kw in query_lower for kw in
               ["head", "director", "officer", "official", "secretary", "administrator", "personnel"]):
            info, src = self._get_officials_info(query_lower, location)
            knowledge_context += info
            sources.extend(src)

        # Check for crop information queries
        if any(kw in query_lower for kw in
               ["palay", "rice", "corn", "mais", "coconut", "sugarcane", "crops", "farming", "cultivation"]):
            info, src = self._get_crop_info(query_lower)
            knowledge_context += info
            sources.extend(src)

        # Check for government programs
        if any(kw in query_lower for kw in
               ["program", "loan", "financing", "subsidy", "support", "assistance", "credit"]):
            info, src = self._get_programs_info()
            knowledge_context += info
            sources.extend(src)

        # Check for pest management
        if any(kw in query_lower for kw in
               ["pest", "disease", "infestation", "control", "spray", "pesticide", "aphid", "borer"]):
            info, src = self._get_pest_management_info(query_lower)
            knowledge_context += info
            sources.extend(src)

        # Check for seasonal information
        if any(kw in query_lower for kw in
               ["season", "wet", "dry", "planting", "harvest", "plant", "plant time", "weather"]):
            info, src = self._get_seasonal_info(query_lower)
            knowledge_context += info
            sources.extend(src)

        return knowledge_context, list(set(sources))

    def _get_officials_info(self, query: str, location: Optional[str] = None) -> Tuple[str, List[str]]:
        """Get official contact information with sources"""
        context = ""
        sources = []

        # National officials
        if any(kw in query for kw in ["secretary", "da", "department of agriculture", "national"]):
            context += "\n=== DEPARTMENT OF AGRICULTURE LEADERSHIP ===\n"
            da = AGRICULTURE_OFFICIALS["National"]["DA"]
            context += f"Secretary: {da['secretary']}\n"
            context += f"Hotline: {da['hotline']}\n"
            context += f"Website: {da['website']}\n"
            context += f"Email: {da['email']}\n"
            sources.append(da['website'])

        # Regional officials
        if location:
            location_lower = location.lower()
            for region, info in AGRICULTURE_OFFICIALS["Regional"].items():
                if any(region_name.lower() in location_lower for region_name in info.get("regions_covered", [])):
                    context += f"\n=== {region} AGRICULTURE OFFICE ===\n"
                    context += f"Office: {info['office']}\n"
                    context += f"Location: {info['location']}\n"
                    context += f"Contact: {info['contact']}\n"
                    context += f"Website: {info.get('website', 'N/A')}\n"
                    if 'website' in info:
                        sources.append(info['website'])
                    break

        # Provincial officials
        if location:
            for province, info in AGRICULTURE_OFFICIALS["Provincial"].items():
                if province.lower() in location.lower():
                    context += f"\n=== {province.upper()} AGRICULTURE ===\n"
                    context += f"Head: {info['pao_head']}\n"
                    context += f"Position: {info['position']}\n"
                    context += f"Office: {info['office']}\n"
                    context += f"Address: {info['address']}\n"
                    context += f"Phone: {info['phone']}\n"
                    context += f"Website: {info.get('website', 'N/A')}\n"
                    if 'website' in info:
                        sources.append(info['website'])

        return context, sources

    def _get_crop_info(self, query: str) -> Tuple[str, List[str]]:
        """Get comprehensive crop information with sources"""
        context = ""
        sources = []

        for crop_name, crop_info in CROP_DATABASE.items():
            if crop_name in query or any(alt in query for alt in crop_info.get("common_names", [])):
                context += f"\n=== {crop_name.upper()} CULTIVATION GUIDE ===\n"
                context += f"Common names: {', '.join(crop_info.get('common_names', []))}\n"
                context += f"Tagalog: {crop_info.get('tagalog', 'N/A')}\n"
                context += f"Wet season: {crop_info.get('season_wet', 'N/A')}\n"
                context += f"Dry season: {crop_info.get('season_dry', 'N/A')}\n"
                context += f"Temperature: {crop_info.get('ideal_temperature', 'N/A')}\n"
                context += f"Water requirement: {crop_info.get('water_requirement', 'N/A')}\n"
                context += f"Soil type: {crop_info.get('soil_type', 'N/A')}\n"
                context += f"pH range: {crop_info.get('ph_range', 'N/A')}\n"
                context += f"Average yield: {crop_info.get('average_yield', 'N/A')}\n"
                context += f"\nMajor producing regions: {', '.join(crop_info.get('major_producing_regions', []))}\n"
                context += f"\nCommon pests: {', '.join(crop_info.get('pests', []))}\n"
                context += f"\nCommon diseases: {', '.join(crop_info.get('diseases', []))}\n"
                context += f"\nBest practices:\n"
                for i, practice in enumerate(crop_info.get('best_practices', []), 1):
                    context += f"  {i}. {practice}\n"

                if 'source' in crop_info:
                    sources.append(crop_info['source'])
                break

        return context, sources

    def _get_programs_info(self) -> Tuple[str, List[str]]:
        """Get government programs information with sources"""
        context = "\n=== GOVERNMENT AGRICULTURAL PROGRAMS ===\n"
        sources = []

        for program in GOVERNMENT_PROGRAMS["DA_Programs"]:
            context += f"\n• {program['name']}\n"
            context += f"  Description: {program.get('description', 'N/A')}\n"
            context += f"  Beneficiaries: {program.get('beneficiaries', 'N/A')}\n"
            if 'source' in program:
                sources.append(program['source'])

        context += "\n=== AGRICULTURAL FINANCING OPTIONS ===\n"
        for finance in GOVERNMENT_PROGRAMS["financing_options"]:
            context += f"\n• {finance['name']}\n"
            context += f"  Type: {finance.get('type', 'N/A')}\n"
            context += f"  Website: {finance.get('website', 'N/A')}\n"
            context += f"  Contact: {finance.get('phone', 'N/A')}\n"
            if 'website' in finance:
                sources.append(finance['website'])

        return context, list(set(sources))

    def _get_pest_management_info(self, query: str) -> Tuple[str, List[str]]:
        """Get pest and disease management information with sources"""
        context = "\n=== PEST & DISEASE MANAGEMENT ===\n"
        sources = []

        for pest_name, pest_info in PEST_DISEASE_DATABASE["major_pests"].items():
            if pest_name in query or any(kw in query for kw in pest_info.get("crops", [])):
                context += f"\n• {pest_name.upper()}\n"
                context += f"  Crops affected: {', '.join(pest_info.get('crops', []))}\n"
                context += f"  Damage: {pest_info.get('damage', 'N/A')}\n"
                context += f"  Identification: {pest_info.get('identification', 'N/A')}\n"
                context += f"  Control methods:\n"
                for method in pest_info.get('control', []):
                    context += f"    - {method}\n"
                if 'source' in pest_info:
                    sources.append(pest_info['source'])

        return context, list(set(sources))

    def _get_seasonal_info(self, query: str) -> Tuple[str, List[str]]:
        """Get seasonal information with sources"""
        context = "\n=== SEASONAL AGRICULTURAL GUIDE ===\n"
        sources = []

        if "wet" in query or "june" in query or "july" in query:
            season = SEASONAL_GUIDE["wet_season"]
            context += f"\nWET SEASON ({season['months']})\n"
            context += f"Temperature: {season['temperature']}\n"
            context += f"Rainfall: {season['rainfall']}\n"
            context += f"Crops: {', '.join(season['crops'])}\n"
            context += f"\nPlanting activities:\n"
            for activity in season["planting_activities"]:
                context += f"  - {activity}\n"
            if 'source' in season:
                sources.append(season['source'])

        if "dry" in query or "december" in query or "march" in query:
            season = SEASONAL_GUIDE["dry_season"]
            context += f"\nDRY SEASON ({season['months']})\n"
            context += f"Temperature: {season['temperature']}\n"
            context += f"Rainfall: {season['rainfall']}\n"
            context += f"Crops: {', '.join(season['crops'])}\n"
            context += f"\nWater management is critical - plan irrigation schedule\n"
            if 'source' in season:
                sources.append(season['source'])

        return context, list(set(sources))

    async def generate_response_with_data(self,
                                          message: str,
                                          location: Optional[str] = None) -> ResponseWithSources:
        """
        Generate response with COMPREHENSIVE HALLUCINATION PREVENTION and SOURCE ATTRIBUTION
        """
        try:
            message_lower = message.lower()
            now = datetime.now()
            current_datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")

            # ===== DETECT PRICE QUERY =====
            price_keywords = ["price", "presyo", "cost", "magkano", "srp", "market value", "benta", "halaga"]
            is_price_query = any(kw in message_lower for kw in price_keywords)

            # ===== KEYWORDS FOR FORCED WEB SEARCH =====
            force_web_search_keywords = [
                "who is", "sino ang", "sino", "head of", "director of",
                "official", "person", "secretary", "administrator", "contact",
                "phone", "address", "hotline", "website", "email"
            ]

            regular_web_search_keywords = [
                "current", "latest", "today", "now", "recent",
                "news", "update", "announcement", "alert"
            ]

            is_about_people = any(kw in message_lower for kw in force_web_search_keywords)
            needs_web_search = is_price_query or is_about_people or any(
                kw in message_lower for kw in regular_web_search_keywords)

            # ===== KNOWLEDGE BASE CONTEXT (PRIMARY) =====
            knowledge_context, kb_sources = self._get_knowledge_base_info(message, location)
            all_sources = kb_sources.copy()

            # ===== WEB SEARCH (SECONDARY - VALIDATION ONLY) =====
            web_context = ""
            web_sources = []

            if needs_web_search:
                # STRICT PHILIPPINE CONTEXT FOR PRICES
                if is_price_query:
                    # Append specific keywords to force PH context
                    search_query = f"latest price of {message} in {location or 'Philippines'} Department of Agriculture Philippines"
                    logger.info(f"PRICE SEARCH (PH ONLY): {search_query}")

                    try:
                        # Force search to bypass any caching checks in search service for prices
                        search_results = await search_service.search(search_query, num_results=5, force=True)

                        if search_results and search_results.get("organic_results"):
                            web_context = "\n=== LATEST MARKET PRICES (WEB VERIFIED) ===\n"
                            for i, result in enumerate(search_results.get("organic_results", [])[:3], 1):
                                web_context += f"{i}. {result.get('title')}\n"
                                web_context += f"   {result.get('snippet')}\n"
                                web_context += f"   Source: {result.get('url')}\n\n"
                                web_sources.append(result.get('url'))
                    except Exception as e:
                        logger.error(f"Price search failed: {str(e)}")

                elif is_about_people:
                    logger.info(f"FORCE WEB SEARCH - People query: {message[:50]}")
                    try:
                        search_results = await search_service.search(message, num_results=5)
                        if search_results and search_results.get("organic_results"):
                            web_context = "\n=== VERIFICATION FROM WEB ===\n"
                            for i, result in enumerate(search_results.get("organic_results", [])[:3], 1):
                                web_context += f"{i}. {result.get('title')}\n"
                                web_context += f"   {result.get('snippet')}\n"
                                web_context += f"   Source: {result.get('url')}\n\n"
                                web_sources.append(result.get('url'))
                    except Exception as e:
                        logger.error(f"People search failed: {str(e)}")

                else:
                    logger.info(f"Standard web search: {message[:50]}")
                    try:
                        search_results = await search_service.search(message, num_results=3)
                        if search_results and search_results.get("organic_results"):
                            web_context = "\n=== LATEST INFORMATION ===\n"
                            for i, result in enumerate(search_results.get("organic_results", [])[:2], 1):
                                web_context += f"{i}. {result.get('title')}\n"
                                web_context += f"   {result.get('snippet')}\n"
                                web_sources.append(result.get('url'))
                    except Exception as e:
                        logger.warning(f"Web search error: {str(e)}")

            all_sources.extend(web_sources)

            # ===== PSA DATA =====
            psa_context = ""
            try:
                if any(kw in message_lower for kw in
                       ["price", "presyo", "production", "area", "yield", "palay", "rice"]):
                    province = "Pangasinan"
                    if location:
                        province = location.split(',')[0].strip()

                    psa_data = await psa_scraper.get_palay_data(province, year=2024)
                    if psa_data:
                        psa_context = f"\n=== OFFICIAL PSA BENCHMARK DATA (2024) ===\n{json.dumps(psa_data, indent=2)}"

                        # Add fallback notification context if detected
                        if "National Fallback" in str(psa_data):
                            psa_context += "\nNOTE: Local data unavailable. Showing National Average."

                        # EXTRACT SPECIFIC URL FROM PSA DATA IF AVAILABLE
                        psa_url = psa_data.get("data", {}).get("source_url")
                        if psa_url:
                            all_sources.append(psa_url)
                        else:
                            all_sources.append(
                                "https://openstat.psa.gov.ph/PXWeb/pxweb/en/DB/DB__2M__NWSNEW/?tablelist=true")

            except Exception as e:
                logger.warning(f"PSA error: {str(e)}")

            # ===== WEATHER DATA =====
            weather_context = ""
            try:
                if any(kw in message_lower for kw in ["weather", "panahon", "forecast", "ulan", "climate"]):
                    weather_data = await weather_service.get_weather(location or "Philippines")
                    if weather_data:
                        weather_context = f"\n=== WEATHER DATA ===\n{json.dumps(weather_data, indent=2)}"

                        # EXTRACT SPECIFIC URL FROM WEATHER DATA IF AVAILABLE
                        weather_url = weather_data.get("source_url")
                        if weather_url:
                            all_sources.append(weather_url)
                        else:
                            all_sources.append("https://bagong.pagasa.dost.gov.ph/weather")

            except Exception as e:
                logger.warning(f"Weather error: {str(e)}")

            # ===== SYSTEM PROMPT CONSTRUCTION =====
            system_prompt = f"""You are AGRI-AID, a FACTUAL Philippine agriculture expert.
CURRENT DATE: {current_datetime_str}

STRICT RULES (PHILIPPINES ONLY):
1. Answer ONLY questions about Philippine agriculture. Refuse others.
2. **PRICE QUERIES:** - You MUST provide a clear summary sentence FIRST. Example: "The current retail price of Red Onion in Metro Manila ranges from ₱120 to ₱140 per kilo as of [Date]."
   - ONLY use data from the provided context (DA, PSA, or Philippine news search results). 
   - Do NOT use international averages or generic data.
   - If local data is missing but National/Metro Manila data is available, explicitly state: "Specific price data for [Location] was unavailable. The national average/Metro Manila price is [Price]."
   - Do NOT just output a list of links. Write a helpful response.
3. **SOURCES:** Use the knowledge base first, then the web search results. 
4. **OFFICIALS:** Only use names/titles listed in the verified knowledge base.

KNOWLEDGE BASE:
{knowledge_context}

{web_context}

{psa_context}

{weather_context}

RESPONSE FORMAT:
- Direct and confident answer in English.
- For prices: Specific range in PHP, Location, and Date.
- No fluff or disclaimers like "As an AI...".
"""

            if location:
                system_prompt += f"\n\nUser Location: {location}"

            logger.info(
                f"Generating response (people={is_about_people}, price={is_price_query}, sources={len(all_sources)})")

            response = self.llm.invoke(system_prompt + f"\n\nQuestion: {message}")

            # Clean response
            unwanted_phrases = [
                "Alternatively",
                "Let me know",
                "how else can",
                "Source: None",
                "Note:",
                "However,"
            ]

            lines = response.split("\n")
            cleaned_lines = [
                line for line in lines
                if not any(phrase in line for phrase in unwanted_phrases)
            ]

            response = "\n".join(cleaned_lines).strip()

            # Remove duplicates from sources
            all_sources = list(set(filter(None, all_sources)))

            return ResponseWithSources(response, all_sources)

        except Exception as e:
            logger.error(f"Error in generate_response_with_data: {str(e)}")
            raise

    async def generate_response(self,
                                message: str,
                                location: Optional[str] = None) -> ResponseWithSources:
        """Generate response with source attribution"""
        try:
            # Get knowledge base info with sources
            knowledge_context, sources = self._get_knowledge_base_info(message, location)
            now = datetime.now()
            current_datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p %Z")

            system_prompt = f"""You are AGRI-AID, a Philippine agricultural expert.
CURRENT DATE: {current_datetime_str}

Help farmers with:
- Crop cultivation techniques
- Pest and disease management
- Seasonal planning
- Government programs
- Market information
- Best practices

KNOWLEDGE BASE:
{knowledge_context}

Be direct, helpful, and practical.
Only use information provided above."""

            prompt = message
            if location:
                prompt += f"\n\nLocation: {location}"

            response = self.llm.invoke(system_prompt + "\n\nUser: " + prompt)

            return ResponseWithSources(response, sources)

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            raise

    async def get_ollama_status(self) -> dict:
        """Check Ollama status"""
        try:
            self.llm.invoke("test")
            return {
                "status": "connected",
                "model": settings.OLLAMA_MODEL,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return {
                "status": "disconnected",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Initialize service
ollama_service = OllamaService()