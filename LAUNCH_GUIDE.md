# 🚀 Quick Start Guide

## Single Command Launch

Start the entire EPL Betting Platform with one command:

```powershell
.\start-app.ps1
```

## Launch Options

```powershell
# Normal startup (recommended)
.\start-app.ps1

# Development mode with detailed logging
.\start-app.ps1 -dev

# Quiet startup with minimal output
.\start-app.ps1 -quiet

# Show help and options
.\start-app.ps1 -help
```

## What Gets Started

✅ **Backend API Server** (FastAPI with comprehensive EPL data)
✅ **Frontend Interface** (HTML/CSS/JS with game week cards)  
✅ **Real Data Integration** (Football-Data.org API)
✅ **ML Prediction Models** (Trained on 1,140+ matches)
✅ **Game Week Reconciliation** (Model vs actual results)

## Access Points

- **Main App**: http://127.0.0.1:8040
- **API Docs**: http://127.0.0.1:8040/docs  
- **Game Weeks**: http://127.0.0.1:8040/#game-weeks

## Features

🎮 **Game Week Cards** - Like NFL betting site with EPL matches
🤖 **Model Reconciliation** - Predictions vs actual results
📊 **Real EPL Data** - 380 fixtures, 20 teams, 38+ players
⚽ **ML Predictions** - Goals, results, confidence scores
📈 **Performance Tracking** - Weekly model accuracy metrics

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the server.

---

*Single launcher replaces all previous launch files for simplicity.*