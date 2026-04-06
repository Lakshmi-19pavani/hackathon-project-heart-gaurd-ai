# Heart Disease Prediction App - Terminal Commands

## To Run the Application:

```bash
cd /home/dell/Downloads/project-hackathon/heart-gaurd-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 backend/app.py
```

## Access in Browser:
```
http://localhost:5000
```

## Production/Deploy Start Command:
```bash
gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

## Login Credentials:
- Username: admin
- Password: admin123

## New Pages:
- `/login` - Login page
- `/dashboard` - Main dashboard
- `/alert-family` - Alert Family Contacts (10 dummy contacts from MongoDB)
- `/call-ambulance` - Call Ambulance (Emergency numbers: 108, 112, 101, etc.)
- `/hospitals` - Nearby Hospitals (10 hospitals from MongoDB)
- `/send-location` - Send Location page

## Test API Commands:

```bash
# Login
curl -c cookies.txt -X POST http://localhost:5000/login -d "username=admin&password=admin123"

# Check database status
curl http://localhost:5000/api/db-status

# Alert family contacts
curl -b cookies.txt -X POST http://localhost:5000/api/emergency -H "Content-Type: application/json" -d '{"action":"alert_family","lat":17.3850,"lng":78.4867}'

# Call ambulance
curl -b cookies.txt -X POST http://localhost:5000/api/emergency -H "Content-Type: application/json" -d '{"action":"call_ambulance","lat":17.3850,"lng":78.4867}'

# Get hospitals
curl -b cookies.txt -X POST http://localhost:5000/api/hospitals -H "Content-Type: application/json" -d '{"lat":17.3850,"lng":78.4867}'
```
