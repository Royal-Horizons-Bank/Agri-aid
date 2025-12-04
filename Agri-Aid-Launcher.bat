@ECHO OFF
REM BFCPEOPTIONSTART
REM Advanced BAT to EXE Converter www.BatToExeConverter.com
REM BFCPEEXE=C:\agriaid-chatbot-v1.1\Agri-Aid-Launcher.exe
REM BFCPEICON=C:\Users\My PC\Downloads\Your paragraph text.ico
REM BFCPEICONINDEX=-1
REM BFCPEEMBEDDISPLAY=0
REM BFCPEEMBEDDELETE=1
REM BFCPEADMINEXE=1
REM BFCPEINVISEXE=0
REM BFCPEVERINCLUDE=1
REM BFCPEVERVERSION=1.2.0.0
REM BFCPEVERPRODUCT=Agri-Aid Launcher
REM BFCPEVERDESC=Launcher for Agri-aid Chatbot.
REM BFCPEVERCOMPANY=R J Salcedo
REM BFCPEVERCOPYRIGHT=Copyright Info
REM BFCPEWINDOWCENTER=1
REM BFCPEDISABLEQE=0
REM BFCPEWINDOWHEIGHT=30
REM BFCPEWINDOWWIDTH=120
REM BFCPEWTITLE=Window Title
REM BFCPEOPTIONEND
@echo off
title AGRI-AID Chatbot Launcher
chcp 65001 > nul

:looplaunch
cls
echo.
echo ███████╗ █████╗ ██╗      ██████╗███████╗██████╗  ██████╗ 
echo ██╔════╝██╔══██╗██║     ██╔════╝██╔════╝██╔══██╗██╔═══██╗
echo ███████╗███████║██║     ██║     █████╗  ██║  ██║██║   ██║
echo ╚════██║██╔══██║██║     ██║     ██╔══╝  ██║  ██║██║   ██║
echo ███████║██║  ██║███████╗╚██████╗███████╗██████╔╝╚██████╔╝
echo ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═════╝  ╚═════╝ 
echo.
echo           ➤ AGRI-AID Multi-Process Launcher v1.0
echo. 
REM Track status with simple flags
set OLLAMASTARTED=0
set BACKENDSTARTED=0
set FRONTENDSTARTED=0

REM ========= START OLLAMA =========
echo [1/3] Starting Ollama...
start "" /min cmd /c "ollama serve"
echo      Ollama launch command sent
timeout /t 2 >nul

REM ========= START BACKEND =========
echo [2/3] Starting Backend (FastAPI)...
start "" /min cmd /c "cd /d C:\agriaid-chatbot-v1.1\backend && call ..\venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 5"
echo      Backend launch command sent
timeout /t 2 >nul

REM ========= START FRONTEND =========
echo [3/3] Starting Frontend (Static Server)...
start "" /min cmd /c "cd /d C:\agriaid-chatbot-v1.1\frontend && python -m http.server 3000 --bind 0.0.0.0"
echo      Frontend launch command sent
timeout /t 3 >nul

REM ========= OPEN INCOGNITO TABS =========
REM ---- edge ----
echo.
echo Opening documentation and site in browser (Incognito)...
REM start "" msedge.exe --inprivate "http://localhost:8000/docs"
REM start "" msedge.exe --inprivate "http://localhost:3000"

REM ========= SHOW MENU =========
:menu
echo.
echo ======================================================
echo         Type the number for an action and press ENTER:
echo ------------------------------------------------------
echo         1. End FastAPI Backend
echo         2. End Frontend Server
echo         3. End Ollama
echo         4. Open Docs and Site (Incognito)
echo         5. End ALL and EXIT
echo         R. RESTART everything
echo         Q. Quit launcher (servers continue running)
echo ------------------------------------------------------
echo ======================================================
set /p action=Your choice: 

if /i "%action%"=="1" (
    echo Killing Backend...
    taskkill /im uvicorn.exe /f >nul 2>&1
    echo Backend process stopped.
    goto menu
)
if /i "%action%"=="2" (
    echo Killing Frontend...
    taskkill /im python.exe /f >nul 2>&1
    echo Frontend process stopped.
    goto menu
)
if /i "%action%"=="3" (
    echo Killing Ollama...
    taskkill /im ollama.exe /f >nul 2>&1
    echo Ollama process stopped.
    goto menu
)
if /i "%action%"=="4" (
    start "" msedge.exe --incognito "http://localhost:8000/docs"
    start "" msedge.exe --incognito "http://localhost:3000"
    echo Reopened browser tabs.
    goto menu
)
if /i "%action%"=="5" (
    echo Killing ALL...
    taskkill /im uvicorn.exe /f >nul 2>&1
    taskkill /im python.exe /f >nul 2>&1
    taskkill /im ollama.exe /f >nul 2>&1
    exit
)
if /i "%action%"=="r" (
    echo Restarting all servers...
    taskkill /im uvicorn.exe /f >nul 2>&1
    taskkill /im python.exe /f >nul 2>&1
    taskkill /im ollama.exe /f >nul 2>&1
    timeout /t 2 >nul
    goto looplaunch
)
if /i "%action%"=="q" (
    echo Launcher window closed. Servers remain active.
    exit
)
echo Invalid option. Please choose 1-5, R, or Q.
goto menu
