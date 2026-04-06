# Heart Guard AI

Flask-based heart risk prediction + emergency support app.

## Local Run
```bash
cd /home/dell/Downloads/project-hackathon/heart-gaurd-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 backend/app.py
```

App URL: `http://localhost:5000`

## Production Run
```bash
gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

## Optional Environment Variables
- `PORT` (default: `5000`)
- `SECRET_KEY` (recommended in production)
- `USE_MONGO` (`true` or `false`, default: `true`)
- `MONGO_URI` (used when `USE_MONGO=true`)
