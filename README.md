# Soccer Betting Platform

A predictable soccer betting platform focused on EPL with scalability for multiple leagues.

## 🎯 Current Status
✅ **Foundation Complete** - The core platform is built and ready for development!

- ✅ FastAPI backend with comprehensive REST endpoints
- ✅ Database models for leagues, teams, players, matches, and predictions  
- ✅ Machine learning prediction framework (heuristic-based, ready for ML models)
- ✅ Comprehensive betting market support
- ✅ Development environment configured
- ✅ Testing framework set up
- ✅ Docker support included

## 🚀 Next Steps for Production
1. **Connect to Real Data**: Integrate with EPL APIs (Football-API, RapidAPI)
2. **Database**: Set up PostgreSQL for production data
3. **Train ML Models**: Use historical match data to train prediction algorithms
4. **Authentication**: Add user authentication and betting features  
5. **Frontend**: Build web interface for placing bets and viewing predictions

## Features
- Total goals prediction
- First half goals prediction  
- Game result prediction (Win/Loss/Draw)
- Player statistics (goals, assists)
- Goalie saves tracking
- Corner statistics (total and team corners)
- Player shots and shots on goal
- Machine learning models for predictions
- API integration for live data
- Historical data storage and analysis

## Technology Stack
- **Backend**: Python with FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **ML/Analytics**: Scikit-learn, TensorFlow, Pandas, NumPy
- **Containerization**: Docker
- **Testing**: pytest
- **API**: RESTful API with real-time data integration

## Project Structure
```
├── app/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   ├── models/            # Database models
│   ├── services/          # Business logic
│   └── ml/                # Machine learning models
├── data/                  # Data storage and processing
├── tests/                 # Test files
├── docker/                # Docker configurations
└── scripts/               # Utility scripts
```

## Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 12+ (optional - SQLite used by default for development)
- Docker (optional)

### Quick Start
1. **Clone and navigate to the project**
2. **Create virtual environment** (already configured)
3. **Install dependencies** (already installed):
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the development server**:
   ```powershell
   # Using PowerShell script (recommended)
   .\start-app.ps1
   
   # Using batch file
   start-app.bat
   
   # Using VS Code task
   Ctrl+Shift+P -> "Tasks: Run Task" -> "Start Soccer Betting API"
   
   # Or directly in terminal
   uvicorn app.main:app --reload
   ```
5. **Visit the API documentation**:
   - Interactive Docs: http://127.0.0.1:8000/docs
   - ReDoc: http://127.0.0.1:8000/redoc

### Script Options
- **PowerShell**: `.\start-app.ps1 -Port 3000 -ServerHost 0.0.0.0 -NoReload`
- **Batch**: Simple double-click to start on default settings

### Database Setup (Optional)
For production use with PostgreSQL:
1. Set up PostgreSQL database
2. Copy `.env.example` to `.env` and configure DATABASE_URL
3. Run migrations: `alembic upgrade head`
4. Seed initial data: `python -m scripts.seed_data`

### Development
- **Run tests**: Use "Run Tests" task or `pytest tests/`
- **Format code**: Use "Format Code (Black)" task or `black .`
- **Lint code**: Use "Lint Code (Flake8)" task
- **Demo API**: `python examples/demo_api.py`

## API Endpoints
- `/api/v1/predictions/` - Prediction endpoints
- `/api/v1/matches/` - Match data
- `/api/v1/players/` - Player statistics
- `/api/v1/teams/` - Team information

## Deploying to Render

This repo includes a Dockerfile and a Render Blueprint (`render.yaml`). To deploy:

1. Push to GitHub (mostgood1/soccer-betting).
2. In Render, create a new Web Service from this repo.
3. Render will detect the Dockerfile. Ensure the following are set:
    - Service type: Web (Docker)
    - Health Check Path: `/`
    - Disk: mount `/app/data` (1GB is sufficient) for week snapshots and reconciliation files.
    - Environment variables:
       - `ODDS_API_KEY` (required for The Odds API H2H odds)
       - `ODDS_API_REGIONS` (e.g., `uk,eu,us`)
       - `ALLOW_ON_DEMAND_PREDICTIONS=1` (optional; allows compute for unlocked weeks)
4. The container runs: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (Render injects `PORT`).
5. Open your service URL to view the frontend and `/docs` for API docs.


## License
MIT License