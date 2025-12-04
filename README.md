🌾 Agri-Aid: Agricultural Intelligence Assistant

Agri-Aid is an advanced, AI-powered agricultural assistant designed specifically for farmers and agricultural stakeholders in the Philippines. It leverages local LLMs (Ollama), computer vision, and real-time data scraping to provide accurate, localized information on crop management, pest control, market prices, and weather forecasts.

🚀 Key Features

🤖 Local AI Chatbot: Powered by Llama 3.1 running locally via Ollama. It provides expert advice on farming techniques, government programs, and seasonal planning without relying on expensive cloud APIs.

📷 Smart Image Analysis: Upload photos of crops or pests for instant identification and diagnosis.

Strict Agronomist Mode: Filters out non-agricultural images (selfies, cars, etc.) to prevent hallucinations.

Pest & Disease Detection: Identifies specific Philippine threats like Rice Black Bug, Tungro, and Armyworm.

💰 Real-Time Market Prices:

Scrapes daily price monitoring data from the Department of Agriculture (DA) and Philippine Statistics Authority (PSA).

Provides localized price updates (e.g., "Price of Red Onion in Nueva Ecija").

National Fallback: Automatically retrieves national averages if specific provincial data is unavailable.

🌦️ Localized Weather:

Fetches real-time weather updates specifically from PAGASA (via Serper search scraping) for accurate, hyper-local forecasts.

📍 Auto-Location: Automatically detects the user's province (via IP) to tailor responses and data (weather/prices) to their specific region.

🔗 Verified Sources: Every response includes clickable citations to official government websites (PhilRice, BPI, DA, PSA) to ensure trust and verification.

🛠️ Tech Stack

Backend

Framework: FastAPI (Python 3.12)

AI Engine: Ollama (Running llama3.1:8b and llava for vision)

Search & Scraping: Serper API (Google Search) + BeautifulSoup4

Image Processing: Pillow (PIL) + Ollama Vision

Asynchronous: Built with asyncio and aiohttp for high performance.

Frontend

Core: HTML5, Vanilla JavaScript (ES6+)

Styling: Tailwind CSS (via CDN)

UI Design: Responsive, mobile-first "Gemini-style" interface with chat bubbles and floating image previews.

Infrastructure

Containerization: Docker & Docker Compose

Windows Launcher: Custom .bat / .exe launcher for one-click startup on Windows machines.

📦 Installation & Setup

Prerequisites


Python 3.12+ installed (for local dev without Docker).

Ollama installed on your host machine (optional, if running outside Docker).


If you have the standalone build:

Double-click Agri-Aid-Launcher.exe.


The browser will open automatically to http://localhost:3000.

