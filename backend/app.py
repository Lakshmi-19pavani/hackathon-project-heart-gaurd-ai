from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import pickle
import numpy as np
import logging
import json
import os
import math
from datetime import datetime

# Initialize Flask app
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
MODEL_PATH = os.path.join(BACKEND_DIR, 'heart_model.pkl')

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, 'templates'),
    static_folder=os.path.join(FRONTEND_DIR, 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'heart_health_guardian_secret_key_2026')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# ==================== MONGODB SETUP ====================
mongo_env = os.environ.get('USE_MONGO')
if mongo_env is None:
    # Safer default for serverless deploys: only enable Mongo when URI is configured.
    MONGO_ENABLED = bool(os.environ.get('MONGO_URI'))
else:
    MONGO_ENABLED = mongo_env.lower() in ('1', 'true', 'yes')
if MONGO_ENABLED:
    try:
        from pymongo import MongoClient
        MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()  # Test connection
        db = client['heartguard_db']
        USE_MONGO = True
        app.logger.info('MongoDB connected successfully.')
    except Exception as e:
        app.logger.warning(f'MongoDB not available ({e}). Using in-memory fallback.')
        USE_MONGO = False
        db = None
else:
    USE_MONGO = False
    db = None
    app.logger.info('MongoDB disabled via USE_MONGO env var. Using in-memory fallback.')

# ==================== SEED DATA ====================

DEFAULT_USERS = {
    'admin': {'password': 'admin123', 'name': 'Admin User'},
    'doctor1': {'password': 'doctor123', 'name': 'Dr. Sharma'},
    'doctor2': {'password': 'doctor123', 'name': 'Dr. Patel'},
    'user': {'password': 'user123', 'name': 'Health User'},
    'demo': {'password': 'demo123', 'name': 'Demo User'},
}

DEFAULT_CONTACTS = [
    {'name': 'Rajesh Kumar', 'phone': '9876500001', 'relationship': 'Parent'},
    {'name': 'Priya Sharma', 'phone': '9876500002', 'relationship': 'Spouse'},
    {'name': 'Dr. Venkat Rao', 'phone': '9876500003', 'relationship': 'Doctor'},
    {'name': 'Amit Patel', 'phone': '9876500004', 'relationship': 'Brother'},
    {'name': 'Sneha Reddy', 'phone': '9876500005', 'relationship': 'Sister'},
    {'name': 'Dr. Srinivas', 'phone': '9876500006', 'relationship': 'Cardiologist'},
    {'name': 'Maheshwar Rao', 'phone': '9876500007', 'relationship': 'Uncle'},
    {'name': 'Lakshmi Devi', 'phone': '9876500008', 'relationship': 'Mother'},
    {'name': 'Anil Kumar', 'phone': '9876500009', 'relationship': 'Cousin'},
    {'name': 'Sunita Sharma', 'phone': '9876500010', 'relationship': 'Aunt'}
]

DEFAULT_HOSPITALS = [
    {'name': 'NIMS Hospital', 'address': 'Punjagutta, Hyderabad', 'phone': '040-23390202', 'lat': 17.4156, 'lng': 78.4424, 'emergency': True, 'speciality': 'Multi-Specialty, Cardiology'},
    {'name': 'Apollo Hospital', 'address': 'Jubilee Hills, Hyderabad', 'phone': '040-23607777', 'lat': 17.4239, 'lng': 78.4107, 'emergency': True, 'speciality': 'Cardiology, Cardiac Surgery'},
    {'name': 'KIMS Hospital', 'address': 'Secunderabad', 'phone': '040-44885000', 'lat': 17.4399, 'lng': 78.4983, 'emergency': True, 'speciality': 'Cardiology, Critical Care'},
    {'name': 'Yashoda Hospital', 'address': 'Somajiguda, Hyderabad', 'phone': '040-45674567', 'lat': 17.4225, 'lng': 78.4569, 'emergency': True, 'speciality': 'Multi-Specialty, Heart Care'},
    {'name': 'Care Hospital', 'address': 'Banjara Hills, Hyderabad', 'phone': '040-30418888', 'lat': 17.4156, 'lng': 78.4347, 'emergency': True, 'speciality': 'Cardiac Sciences'},
    {'name': 'Gandhi Hospital', 'address': 'Musheerabad, Hyderabad', 'phone': '040-27505566', 'lat': 17.4062, 'lng': 78.4802, 'emergency': True, 'speciality': 'General, Emergency Care'},
    {'name': 'Osmania General Hospital', 'address': 'Afzalgunj, Hyderabad', 'phone': '040-24600146', 'lat': 17.3762, 'lng': 78.4720, 'emergency': True, 'speciality': 'General, Critical Care'},
    {'name': 'Star Hospitals', 'address': 'Banjara Hills, Hyderabad', 'phone': '040-44777777', 'lat': 17.4148, 'lng': 78.4353, 'emergency': True, 'speciality': 'Heart Institute'},
    {'name': 'Continental Hospital', 'address': 'Gachibowli, Hyderabad', 'phone': '040-67000000', 'lat': 17.4401, 'lng': 78.3489, 'emergency': True, 'speciality': 'Multi-Specialty, Cardiology'},
    {'name': 'AIG Hospitals', 'address': 'Gachibowli, Hyderabad', 'phone': '040-42444222', 'lat': 17.4326, 'lng': 78.3564, 'emergency': True, 'speciality': 'Gastro, Cardiology'},
]

# In-memory fallback stores
MEM_USERS = dict(DEFAULT_USERS)
MEM_CONTACTS = {}
MEM_HOSPITALS = list(DEFAULT_HOSPITALS)
MEM_EMERGENCY_LOGS = []
MEM_EQUIPMENT_RENTALS = []
MEM_HEALTH_SERVICES = [
    {
        'username': 'demo',
        'service': 'Home Nursing Care',
        'nurse_name': 'Nurse Sarah Parker',
        'date': '2026-03-18',
        'time': '10:00 AM',
        'status': 'confirmed',
        'duration': '4 hours',
        'cost': '₹2000'
    },
    {
        'username': 'user',
        'service': 'Post-Surgery Care',
        'nurse_name': 'Nurse Mark Stevenson',
        'date': '2026-03-19',
        'time': '09:00 AM',
        'status': 'pending',
        'duration': '8 hours',
        'cost': '₹3500'
    }
]
MEM_HEALTH_WORKERS = []
MEM_LAB_TESTS = [
    {
        'username': 'demo',
        'test_name': 'Lipid Profile',
        'date': '2026-03-18',
        'time': '08:00 AM',
        'status': 'completed',
        'result': 'Normal',
        'lab': 'Apollo Diagnostics'
    },
    {
        'username': 'user',
        'test_name': 'ECG (Electrocardiogram)',
        'date': '2026-03-20',
        'time': '10:00 AM',
        'status': 'pending',
        'result': 'Awaited',
        'lab': 'NIMS Lab'
    },
    {
        'username': 'demo',
        'test_name': 'Cardiac Enzymes',
        'date': '2026-03-15',
        'time': '09:00 AM',
        'status': 'completed',
        'result': 'Normal',
        'lab': 'KIMS Lab'
    }
]
MEM_EQUIPMENT_RENTALS = []
MEM_LOGINS = []
MEM_PREDICTIONS = []
MEM_NURSES = [
    {'name': 'Nurse Sarah Parker', 'rating': '4.9', 'specialization': 'Post-Op Care', 'status': 'verified', 'id': 'N001'},
    {'name': 'Nurse Mark Stevenson', 'rating': '4.8', 'specialization': 'Home ICU', 'status': 'verified', 'id': 'N002'},
    {'name': 'Nurse Jessica Moore', 'rating': '0.0', 'specialization': 'Pediatric Care', 'status': 'pending', 'id': 'N003'}
]

# Seed Initial Cardiologist Appointments for Demo
SEED_APPOINTMENTS = [
    {
        'username': 'demo',
        'patient_name': 'Ramesh Chandra',
        'phone': '9988776655',
        'specialization': 'Cardiologist',
        'hospital': 'NIMS Hospital',
        'doctor': 'Dr. Arun Kumar',
        'doctor_username': 'doctor1',
        'time': '10:30 AM',
        'date': '2026-03-20',
        'reason': 'Frequent chest pain and shortness of breath',
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    },
    {
        'username': 'user',
        'patient_name': 'Lata M.',
        'phone': '9440011223',
        'specialization': 'Cardiologist',
        'hospital': 'Apollo Hospital',
        'doctor': 'Dr. Meera Reddy',
        'doctor_username': 'doctor2',
        'time': '02:00 PM',
        'date': '2026-03-22',
        'reason': 'Post-surgery follow-up checkup',
        'status': 'pending',
        'created_at': datetime.now().isoformat()
    }
]
MEM_APPOINTMENTS = list(SEED_APPOINTMENTS)


def seed_mongodb():
    """Seed MongoDB with default data if collections are empty."""
    if not USE_MONGO:
        return
    # Seed users
    if db.users.count_documents({}) == 0:
        for uname, udata in DEFAULT_USERS.items():
            db.users.insert_one({'username': uname, **udata})
    else:
        # Ensure doctor users exist
        for uname, udata in DEFAULT_USERS.items():
            if uname.startswith('doctor') and not db.users.find_one({'username': uname}):
                db.users.insert_one({'username': uname, **udata})
        app.logger.info('  Seeded default users')
    # Seed hospitals
    if db.hospitals.count_documents({}) == 0:
        db.hospitals.insert_many(DEFAULT_HOSPITALS)
        app.logger.info('  Seeded default hospitals')
    # Seed appointments
    if db.appointments.count_documents({}) == 0:
        db.appointments.insert_many(SEED_APPOINTMENTS)
        app.logger.info('  Seeded sample appointments')
    else:
        # Update existing appointments that don't have doctor_username
        doctor_mapping = {
            'Dr. Arun Kumar': 'doctor1',
            'Dr. Sarah Varma': 'doctor1',
            'Dr. Vikram Seth': 'doctor2',
            'Dr. Meera Reddy': 'doctor2',
            'Dr. Sanjay Gupta': 'doctor1',
            'Dr. Anil Shah': 'doctor2',
            'Dr. Rajesh Sharma': 'doctor1',
            'Dr. Priya Das': 'doctor2',
            'Dr. N. Rao': 'doctor1',
            'Dr. K. Smitha': 'doctor2',
            'Dr. Harsha Vardhan': 'doctor1',
            'Dr. S. Srinivas': 'doctor2',
            'Dr. Kavitha Reddy': 'doctor1',
            'Dr. Manoj Tiwari': 'doctor2'
        }
        for doctor_name, doctor_username in doctor_mapping.items():
            db.appointments.update_many(
                {'doctor': doctor_name, 'doctor_username': {'$exists': False}},
                {'$set': {'doctor_username': doctor_username}}
            )
        app.logger.info('  Updated appointments with doctor usernames')
    # Seed nurses
    if db.nurses.count_documents({}) == 0:
        db.nurses.insert_many(MEM_NURSES)
        app.logger.info('  Seeded nursing staff')
    # Seed health services
    # (Disabled to ensure only real user bookings are shown)
    # if db.health_services.count_documents({}) == 0:
    #     db.health_services.insert_many(MEM_HEALTH_SERVICES)
    #     app.logger.info('  Seeded sample health services')
    app.logger.info('✅ MongoDB seed complete')


if USE_MONGO:
    seed_mongodb()


# ==================== DB HELPER FUNCTIONS ====================

def db_get_user(username):
    """Get user from MongoDB or memory."""
    if USE_MONGO:
        user = db.users.find_one({'username': username}, {'_id': 0})
        return user
    return MEM_USERS.get(username)


def db_create_user(username, password, name):
    """Create user in MongoDB or memory."""
    if USE_MONGO:
        if db.users.find_one({'username': username}):
            return False
        db.users.insert_one({'username': username, 'password': password, 'name': name})
        return True
    if username in MEM_USERS:
        return False
    MEM_USERS[username] = {'password': password, 'name': name}
    return True


def db_get_contacts(username):
    """Get contacts for a user from MongoDB or memory."""
    if USE_MONGO:
        contacts = list(db.contacts.find({'username': username}, {'_id': 0, 'username': 0}))
        if not contacts:
            # Auto-seed default dummy contacts for new users
            for i, c in enumerate(DEFAULT_CONTACTS):
                doc = {'username': username, 'contact_id': i + 1, **c}
                db.contacts.insert_one(doc)
            contacts = list(db.contacts.find({'username': username}, {'_id': 0, 'username': 0}))
        return contacts
    # In-memory fallback
    if username not in MEM_CONTACTS:
        import copy
        MEM_CONTACTS[username] = [{'contact_id': i+1, **c} for i, c in enumerate(copy.deepcopy(DEFAULT_CONTACTS))]
    return MEM_CONTACTS.get(username, [])


def db_add_contact(username, name, phone, relationship):
    """Add a contact for a user."""
    if USE_MONGO:
        # Get next ID
        last = db.contacts.find_one({'username': username}, sort=[('contact_id', -1)])
        next_id = (last['contact_id'] + 1) if last else 1
        contact = {'username': username, 'contact_id': next_id, 'name': name, 'phone': phone, 'relationship': relationship}
        db.contacts.insert_one(contact)
        return {'contact_id': next_id, 'name': name, 'phone': phone, 'relationship': relationship}
    # In-memory fallback
    if username not in MEM_CONTACTS:
        MEM_CONTACTS[username] = []
    next_id = len(MEM_CONTACTS[username]) + 1
    contact = {'contact_id': next_id, 'name': name, 'phone': phone, 'relationship': relationship}
    MEM_CONTACTS[username].append(contact)
    return contact


def db_delete_contact(username, contact_id):
    """Delete a contact."""
    if USE_MONGO:
        db.contacts.delete_one({'username': username, 'contact_id': contact_id})
    else:
        if username in MEM_CONTACTS:
            MEM_CONTACTS[username] = [c for c in MEM_CONTACTS[username] if c['contact_id'] != contact_id]


def db_get_hospitals():
    """Get all hospitals from MongoDB or memory."""
    if USE_MONGO:
        return list(db.hospitals.find({}, {'_id': 0}))
    return list(MEM_HOSPITALS)


def db_log_emergency(username, action, lat, lng, contacts_notified):
    """Log emergency action to MongoDB."""
    log = {
        'username': username,
        'action': action,
        'lat': lat,
        'lng': lng,
        'contacts_notified': contacts_notified,
        'timestamp': datetime.utcnow().isoformat()
    }
    if USE_MONGO:
        db.emergency_logs.insert_one(log)
    else:
        MEM_EMERGENCY_LOGS.append(log)
    return log


# ==================== LOAD ML MODEL ====================

# Load the pre-trained model with error handling
try:
    app.logger.debug("Attempting to load the model...")
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    app.logger.debug("Model loaded successfully.")
except FileNotFoundError:
    app.logger.error(f"The model file was not found at {MODEL_PATH}.")
    model = None
except Exception as e:
    app.logger.error(f"An unexpected error occurred: {e}")
    model = None

# ==================== ROUTES ====================

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        selected_role = request.form.get('role', 'user').strip()

        user = db_get_user(username)
        if user and user['password'] == password:
            session['user'] = username
            session['name'] = user['name']
            
            # Track Login
            login_event = {
                'username': username,
                'name': user['name'],
                'timestamp': datetime.now().isoformat(),
                'ip': request.remote_addr
            }
            if USE_MONGO:
                db.login_logs.insert_one(login_event)
            else:
                MEM_LOGINS.append(login_event)

            # Redirect based on selected role or username
            if selected_role == 'admin' or username == 'admin':
                return redirect(url_for('admin_dashboard'))
            # Check if doctor user (username starts with 'doctor')
            elif selected_role == 'doctor' or username.startswith('doctor'):
                return redirect(url_for('doctor_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username or password')

    return render_template('login.html')


@app.route('/admin-dashboard')
def admin_dashboard():
    if 'user' not in session or session.get('user') != 'admin':
        return redirect(url_for('login'))
    
    # Fetch Data
    if USE_MONGO:
        users = list(db.users.find({}, {'_id': 0, 'password': 0}))
        # Filter out seed users for a "real" count if desired, but here we just fetch all
        logins = list(db.login_logs.find({}, {'_id': 0}).sort('timestamp', -1))
        appointments = list(db.appointments.find({}, {'_id': 0}))
        health_services = list(db.health_services.find({}, {'_id': 0}))
        payments = list(db.payments.find({}, {'_id': 0}))
        health_workers = list(db.health_workers.find({}, {'_id': 0}))
        lab_tests = list(db.lab_tests.find({}, {'_id': 0}))
        equipment_rentals = list(db.equipment_rentals.find({}, {'_id': 0}))
        nurses = list(db.nurses.find({}, {'_id': 0}))
        history = list(db.predictions.find({}, {'_id': 0}).sort('timestamp', -1))
    else:
        users = [{'username': u, 'name': d['name']} for u, d in MEM_USERS.items()]
        logins = sorted(MEM_LOGINS, key=lambda x: x['timestamp'], reverse=True)
        appointments = MEM_APPOINTMENTS
        health_services = MEM_HEALTH_SERVICES
        health_workers = MEM_HEALTH_WORKERS
        lab_tests = MEM_LAB_TESTS
        equipment_rentals = MEM_EQUIPMENT_RENTALS
        nurses = MEM_NURSES
        history = sorted(MEM_PREDICTIONS, key=lambda x: x.get('timestamp', ''), reverse=True)
        all_payments = []
        for user_payments in MEM_PAYMENTS.values():
            all_payments.extend(user_payments)
        payments = all_payments
    
    # Define all specializations
    specializations = ['Cardiologist', 'General Doctor', 'Neurologist', 'Orthopedic', 'Pediatrician', 'Dermatologist']
    
    # Process each specialization independently
    spec_registries = {}
    spec_stats = {}
    
    for spec in specializations:
        # 1. Filter raw appointments
        raw_list = [a for a in appointments if a.get('specialization') == spec]
        
        # 2. Filter for "Real" stats (exclude seeds & demo)
        # Seed patients for Cardio are Ramesh and Lata.
        real_list = [
            a for a in raw_list 
            if a.get('patient_name') not in ['Ramesh Chandra', 'Lata M.']
            and a.get('username') not in ['admin', 'demo']
            and a.get('appointment_id')
        ]
        
        # 3. Calculate Stats
        unique_names = set([a.get('patient_name') for a in real_list if a.get('patient_name')])
        spec_stats[spec] = {
            'registrations': len(unique_names),
            'pending': len([a for a in real_list if a.get('status') == 'pending']),
            'verified': len([a for a in real_list if a.get('status') == 'verified'])
        }
        
        # 4. Handle Registry (Add AI Detections only for Cardio)
        registry_list = list(raw_list)
        if spec == 'Cardiologist':
            for record in history:
                if record.get('risk_level') in ['high', 'medium']:
                    risk_entry = {
                        'username': record.get('username'),
                        'patient_name': record.get('username').title(),
                        'phone': '-',
                        'specialization': 'Cardiologist',
                        'hospital': 'AI System Detection',
                        'doctor': 'Risk level: ' + record.get('risk_level').upper(),
                        'reason': f"Detected {record.get('risk_percentage')}% heart risk",
                        'date': record.get('timestamp', '').split('T')[0],
                        'time': record.get('timestamp', '').split('T')[1][:5] if 'T' in record.get('timestamp', '') else '--:--',
                        'status': 'AI_FLAGGED'
                    }
                    if not any(a['username'] == record.get('username') and a['status'] == 'booked' for a in appointments):
                        registry_list.append(risk_entry)
        
        spec_registries[spec] = registry_list

    # Home Service Stats
    h_serv_real = [s for s in health_services if s.get('username') != 'admin']
    h_serv_stats = {
        'registrations': len(h_serv_real),
        'pending': len([s for s in h_serv_real if s.get('status') == 'pending']),
        'verified': len([s for s in h_serv_real if s.get('status') == 'verified'])
    }

    # Lab Test Stats
    lab_real = [l for l in lab_tests if l.get('username') != 'admin']
    lab_stats = {
        'registrations': len(lab_real),
        'pending': len([l for l in lab_real if l.get('status') == 'pending']),
        'verified': len([l for l in lab_real if l.get('status') == 'verified'])
    }

    # Equipment Rental Stats
    equip_real = [e for e in equipment_rentals if e.get('username') != 'admin']
    equip_stats = {
        'registrations': len(equip_real),
        'pending': len([e for e in equip_real if e.get('status') == 'pending']),
        'verified': len([e for e in equip_real if e.get('status') == 'verified'])
    }

    # Nurse Capacity Stats (Workforce)
    nurse_capacity_stats = {
        'total': len(nurses),
        'pending': len([n for n in nurses if n.get('status') == 'pending']),
        'verified': len([n for n in nurses if n.get('status') == 'verified'])
    }

    # Overall Global Stats for the main board
    global_stats = {
        'total_registrations': len(users),
        'total_logins': len([l for l in logins if l['username'] not in ['admin', 'demo']]),
        'total_consultations': len(appointments)
    }

    return render_template('admin_dashboard.html', 
                           users=users, 
                           logins=logins[:10],
                           spec_registries=spec_registries,
                           spec_stats=spec_stats,
                           global_stats=global_stats,
                           h_serv_stats=h_serv_stats,
                           lab_stats=lab_stats,
                           equip_stats=equip_stats,
                           nurse_stats=nurse_capacity_stats,
                           nurses=nurses,
                           health_services=health_services,
                           health_workers=health_workers,
                           lab_tests=lab_tests,
                           equipment_rentals=equipment_rentals,
                           payments=payments)


@app.route('/doctor-dashboard')
def doctor_dashboard():
    if 'user' not in session or not session.get('user', '').startswith('doctor'):
        return redirect(url_for('login'))
    
    username = session['user']
    doctor_name = session.get('name', 'Doctor')
    
    # Fetch health-related data for the doctor
    if USE_MONGO:
        # Get appointments assigned to this doctor (always filter by doctor_username)
        appointments = list(db.appointments.find({'doctor_username': username}, {'_id': 0}))
        
        # Get list of patients (usernames) who have appointments with this doctor
        patient_usernames = set(a.get('username') for a in appointments if a.get('username'))
        
        # Only show data for patients who have appointments with this doctor
        if patient_usernames:
            health_services = list(db.health_services.find({'username': {'$in': list(patient_usernames)}}, {'_id': 0}))
            lab_tests = list(db.lab_tests.find({'username': {'$in': list(patient_usernames)}}, {'_id': 0}))
            raw_medicines = list(db.medicines.find({'username': {'$in': list(patient_usernames)}}, {'_id': 0}))
        else:
            health_services = []
            lab_tests = []
            raw_medicines = []
        
        # Map 'name' to 'medicine_name' for template compatibility
        medicine_reminders = []
        for m in raw_medicines:
            med = dict(m)
            if 'name' in med and 'medicine_name' not in med:
                med['medicine_name'] = med.pop('name')
            medicine_reminders.append(med)
    else:
        # Filter appointments for this specific doctor
        appointments = [a for a in MEM_APPOINTMENTS if a.get('doctor_username') == username]
        # Get list of patients (usernames) who have appointments with this doctor
        patient_usernames = set(a.get('username') for a in appointments if a.get('username'))
        
        # Only show data for patients who have appointments with this doctor
        if patient_usernames:
            # Get health services for this doctor's patients
            health_services = [s for s in MEM_HEALTH_SERVICES if s.get('username') in patient_usernames]
            # Get lab tests for this doctor's patients
            lab_tests = [t for t in MEM_LAB_TESTS if t.get('username') in patient_usernames]
            # Get medicine reminders for this doctor's patients
            medicine_reminders = []
            for patient in patient_usernames:
                if patient in MEM_MEDICINES:
                    medicine_reminders.extend(MEM_MEDICINES[patient])
        else:
            health_services = []
            lab_tests = []
            medicine_reminders = []
    
    # Use the filtered appointments (not all_appointments)
    all_appointments = appointments
    
    return render_template('doctor_dashboard.html',
                           doctor_name=doctor_name,
                           username=username,
                           appointments=appointments,
                           all_appointments=all_appointments,
                           health_services=health_services,
                           lab_tests=lab_tests,
                           medicine_reminders=medicine_reminders)
@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()
    name = request.form.get('name', '').strip()

    if not username or not password or not name:
        return render_template('login.html', error='All fields are required', show_register=True)

    if not db_create_user(username, password, name):
        return render_template('login.html', error='Username already exists', show_register=True)

    session['user'] = username
    session['name'] = name
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# Family Member Login
@app.route('/family-login', methods=['GET', 'POST'])
def family_login():
    if request.method == 'POST':
        family_name = request.form.get('family_name', '').strip()
        patient_name = request.form.get('patient_name', '').strip()
        relationship = request.form.get('relationship', '').strip()
        phone = request.form.get('phone', '').strip()
        
        if family_name and patient_name and relationship:
            session['family_user'] = family_name
            session['patient_name'] = patient_name
            session['relationship'] = relationship
            session['phone'] = phone
            return redirect(url_for('family_dashboard'))
        else:
            return render_template('family_login.html', error='All fields are required')
    
    return render_template('family_login.html')


@app.route('/family-logout')
def family_logout():
    session.clear()
    return redirect(url_for('family_login'))


@app.route('/family-dashboard')
def family_dashboard():
    if 'family_user' not in session:
        return redirect(url_for('family_login'))
    return render_template('family_member_dashboard.html', 
                         family_name=session.get('family_user', 'Family'),
                         patient_name=session.get('patient_name', 'Patient'),
                         relationship=session.get('relationship', 'Family'))


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', name=session.get('name', 'User'))


@app.route('/send-location')
def send_location():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('send_location.html', name=session.get('name', 'User'))


@app.route('/alert-family')
def alert_family():
    """Show dedicated page for alerting family contacts."""
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    user_name = session.get('name', 'User')
    contacts = db_get_contacts(user)
    return render_template('alert_family.html', name=user_name, contacts=contacts)


@app.route('/call-ambulance')
def call_ambulance():
    """Show dedicated page for calling ambulance."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('call_ambulance.html', name=session.get('name', 'User'))


@app.route('/hospitals')
def hospitals():
    """Show dedicated page for nearby hospitals."""
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    all_hospitals = db_get_hospitals()
    return render_template('hospitals.html', name=session.get('name', 'User'), hospitals=all_hospitals)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400

        # Get values and validate they are numbers
        try:
            age = int(data.get('age', 0))
            weight = float(data.get('weight', 0))
            bp_systolic = int(data.get('bp_systolic', 0))
            bp_diastolic = int(data.get('bp_diastolic', 0))
            heart_rate = int(data.get('heart_rate', 0))
            cholesterol = int(data.get('cholesterol', 0))
        except (ValueError, TypeError):
            return jsonify({'error': 'Please enter valid numbers for all health metrics'}), 400
        
        # Validate ranges
        if age < 1 or age > 120:
            return jsonify({'error': 'Age must be between 1 and 120 years'}), 400
        if weight < 20 or weight > 300:
            return jsonify({'error': 'Weight must be between 20 and 300 kg'}), 400
        if bp_systolic < 70 or bp_systolic > 250:
            return jsonify({'error': 'BP Systolic must be between 70 and 250 mmHg'}), 400
        if bp_diastolic < 40 or bp_diastolic > 150:
            return jsonify({'error': 'BP Diastolic must be between 40 and 150 mmHg'}), 400
        if heart_rate < 30 or heart_rate > 220:
            return jsonify({'error': 'Heart Rate must be between 30 and 220 bpm'}), 400
        if cholesterol < 100 or cholesterol > 500:
            return jsonify({'error': 'Cholesterol must be between 100 and 500 mg/dL'}), 400

        diabetes = data.get('diabetes', 'no')
        smoking = data.get('smoking', 'no')
        physical_activity = data.get('physical_activity', 'moderate')

        # Calculate risk score based on health parameters
        risk_score = 0

        # Age risk
        if age > 55:
            risk_score += 3
        elif age > 45:
            risk_score += 2
        elif age > 35:
            risk_score += 1

        # BP risk
        if bp_systolic > 160 or bp_diastolic > 100:
            risk_score += 3
        elif bp_systolic > 140 or bp_diastolic > 90:
            risk_score += 2
        elif bp_systolic > 130 or bp_diastolic > 85:
            risk_score += 1

        # Heart rate risk
        if heart_rate > 100 or heart_rate < 50:
            risk_score += 2
        elif heart_rate > 90 or heart_rate < 55:
            risk_score += 1

        # Cholesterol risk
        if cholesterol > 280:
            risk_score += 3
        elif cholesterol > 240:
            risk_score += 2
        elif cholesterol > 200:
            risk_score += 1

        # Diabetes risk
        if diabetes == 'yes':
            risk_score += 2

        # Smoking risk
        if smoking == 'yes':
            risk_score += 2

        # Physical activity (lower activity = higher risk)
        activity_map = {'sedentary': 3, 'light': 2, 'moderate': 1, 'active': 0, 'very_active': 0}
        risk_score += activity_map.get(physical_activity, 1)

        # Weight/BMI consideration (rough estimate)
        if weight > 100:
            risk_score += 2
        elif weight > 85:
            risk_score += 1

        # Try model prediction if available
        model_prediction = None
        model_probability = 0.0
        if model is not None:
            try:
                # Map to model's expected features: [age, sex, cp, trestbps, chol]
                sex = 1  # default
                cp = min(3, max(0, risk_score // 4))
                input_data = np.array([[age, sex, cp, bp_systolic, cholesterol]])

                # Use predict_proba for a nuanced probability instead of binary predict
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(input_data)[0]
                    model_probability = float(probabilities[1])  # probability of disease
                    model_prediction = 1 if model_probability >= 0.5 else 0
                else:
                    model_prediction = int(model.predict(input_data)[0])
                    model_probability = 1.0 if model_prediction == 1 else 0.0
            except Exception as e:
                app.logger.error(f"Model prediction error: {e}")

        # Determine risk level — blend rule-based score with model probability
        max_possible = 18
        rule_based_pct = min(100, int((risk_score / max_possible) * 100))

        if model_probability > 0:
            # Weighted blend: 70% rule-based (which uses actual inputs), 30% model
            risk_percentage = min(100, int(rule_based_pct * 0.7 + model_probability * 100 * 0.3))
        else:
            risk_percentage = rule_based_pct

        if risk_percentage <= 25:
            risk_level = 'low'
        elif risk_percentage <= 55:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        # Record this prediction for admin review
        prediction_record = {
            'username': session['user'],
            'risk_level': risk_level,
            'risk_percentage': risk_percentage,
            'risk_score': risk_score,
            'metrics': {
                'age': age,
                'bp': f"{bp_systolic}/{bp_diastolic}",
                'cholesterol': cholesterol,
                'heart_rate': heart_rate
            },
            'timestamp': datetime.now().isoformat()
        }

        if USE_MONGO:
            db.predictions.insert_one(prediction_record)
        else:
            MEM_PREDICTIONS.append(prediction_record)

        # Generate warnings
        warnings = []
        if bp_systolic > 140 or bp_diastolic > 90:
            warnings.append({'type': 'danger', 'message': f'⚠️ High Blood Pressure detected: {bp_systolic}/{bp_diastolic} mmHg. Consult a doctor immediately.'})
        elif bp_systolic > 130 or bp_diastolic > 85:
            warnings.append({'type': 'warning', 'message': f'⚡ Elevated Blood Pressure: {bp_systolic}/{bp_diastolic} mmHg. Monitor closely.'})

        if heart_rate > 100:
            warnings.append({'type': 'danger', 'message': f'⚠️ Tachycardia detected: {heart_rate} bpm. Seek medical attention.'})
        elif heart_rate < 50:
            warnings.append({'type': 'danger', 'message': f'⚠️ Bradycardia detected: {heart_rate} bpm. Seek medical attention.'})

        if cholesterol > 240:
            warnings.append({'type': 'danger', 'message': f'⚠️ High Cholesterol: {cholesterol} mg/dL. Immediate lifestyle changes needed.'})
        elif cholesterol > 200:
            warnings.append({'type': 'warning', 'message': f'⚡ Borderline High Cholesterol: {cholesterol} mg/dL. Watch your diet.'})

        if diabetes == 'yes':
            warnings.append({'type': 'warning', 'message': '⚡ Diabetes increases cardiovascular risk. Regular monitoring recommended.'})

        if smoking == 'yes':
            warnings.append({'type': 'danger', 'message': '⚠️ Smoking significantly increases heart disease risk. Consider quitting.'})

        if physical_activity == 'sedentary':
            warnings.append({'type': 'warning', 'message': '⚡ Sedentary lifestyle detected. Aim for at least 150 min of exercise per week.'})

        # Generate prevention guidance
        diet_suggestions = []
        exercise_plans = []
        lifestyle_improvements = []

        if risk_level == 'high':
            diet_suggestions = [
                '🥗 Follow a strict DASH diet (Dietary Approaches to Stop Hypertension)',
                '🐟 Eat omega-3 rich fish (salmon, mackerel) at least 3 times/week',
                '🚫 Eliminate processed foods, trans fats, and excess sodium',
                '🥬 Increase fiber intake to 30g/day with whole grains and vegetables',
                '🫒 Use olive oil instead of butter or vegetable oils',
                '🍎 Eat 5+ servings of fruits and vegetables daily'
            ]
            exercise_plans = [
                '🏥 Consult your doctor before starting any exercise program',
                '🚶 Start with 10-minute walks, gradually increase to 30 minutes',
                '🧘 Practice gentle yoga or tai chi for stress reduction',
                '🏊 Try low-impact activities like swimming or cycling',
                '📅 Aim for 150 minutes of moderate activity per week'
            ]
            lifestyle_improvements = [
                '🚭 Quit smoking immediately — seek professional help',
                '😴 Get 7-8 hours of quality sleep every night',
                '🧘 Practice daily meditation or deep breathing (10 min/day)',
                '📋 Schedule regular health check-ups every 3 months',
                '💊 Take prescribed medications consistently',
                '📱 Use a health monitoring app to track vitals daily'
            ]
        elif risk_level == 'medium':
            diet_suggestions = [
                '🥗 Follow a Mediterranean-style diet rich in vegetables',
                '🐟 Include fatty fish in your meals 2-3 times per week',
                '🧂 Reduce sodium intake to less than 2,300mg/day',
                '🥑 Include heart-healthy fats (avocado, nuts, seeds)',
                '🍵 Replace sugary drinks with green tea or water'
            ]
            exercise_plans = [
                '🏃 30 minutes of brisk walking 5 days a week',
                '🏋️ Add light strength training 2 days per week',
                '🧘 Include flexibility exercises and yoga',
                '🚴 Try cycling or swimming for cardio variety',
                '📊 Track your steps — aim for 8,000-10,000 daily'
            ]
            lifestyle_improvements = [
                '😴 Maintain a consistent sleep schedule (7-8 hours)',
                '🧘 Practice stress management techniques',
                '🚭 If you smoke, create a quit plan now',
                '📋 Get health screenings every 6 months',
                '🤝 Join a health support group or community'
            ]
        else:
            diet_suggestions = [
                '🥗 Maintain a balanced diet with plenty of fruits and vegetables',
                '🐟 Include fish and lean proteins regularly',
                '💧 Stay hydrated — drink 8+ glasses of water daily',
                '🥜 Snack on nuts, seeds, and fresh fruits',
                '🌾 Choose whole grains over refined grains'
            ]
            exercise_plans = [
                '🏃 Stay active with 30+ minutes of exercise daily',
                '🏋️ Include a mix of cardio and strength training',
                '🧘 Practice yoga or stretching for flexibility',
                '🚴 Explore fun activities like dancing, hiking, or sports',
                '📊 Challenge yourself with fitness goals regularly'
            ]
            lifestyle_improvements = [
                '😴 Continue prioritizing quality sleep',
                '🧘 Maintain your stress management routine',
                '📋 Annual health check-ups are recommended',
                '🤝 Stay socially active and connected',
                '📚 Stay informed about heart health best practices'
            ]

        return jsonify({
            'risk_level': risk_level,
            'risk_percentage': risk_percentage,
            'risk_score': risk_score,
            'warnings': warnings,
            'model_prediction': model_prediction,
            'diet_suggestions': diet_suggestions,
            'exercise_plans': exercise_plans,
            'lifestyle_improvements': lifestyle_improvements
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


# ==================== FAMILY CONTACTS (MongoDB) ====================

@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all family contacts for current user from MongoDB."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    user = session['user']
    try:
        contacts = db_get_contacts(user)
        return jsonify({'contacts': contacts})
    except Exception as e:
        return jsonify({'error': f'Failed to load contacts: {str(e)}', 'contacts': []}), 500


@app.route('/api/contacts', methods=['POST'])
def add_contact():
    """Add a family contact to MongoDB."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    user = session['user']
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid request data. Please provide valid JSON.'}), 400

    name = data.get('name', '').strip() if data.get('name') else ''
    phone = data.get('phone', '').strip() if data.get('phone') else ''
    relationship = data.get('relationship', '').strip() if data.get('relationship') else ''

    if not name or not phone:
        return jsonify({'error': 'Name and phone number are required'}), 400

    contact = db_add_contact(user, name, phone, relationship or 'Family Member')
    return jsonify({'success': True, 'contact': contact, 'message': f'✅ {name} added as emergency contact.'})


@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    """Delete a family contact from MongoDB."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    user = session['user']
    db_delete_contact(user, contact_id)
    return jsonify({'success': True, 'message': 'Contact removed.'})


# ==================== HOSPITALS (MongoDB) ====================

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate distance between two GPS coordinates in km."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return round(R * c, 1)


@app.route('/api/hospitals', methods=['POST'])
def get_hospitals():
    """Return nearby hospitals from MongoDB, sorted by distance."""
    data = request.get_json()
    lat = float(data.get('lat', 17.3850))
    lng = float(data.get('lng', 78.4867))

    all_hospitals = db_get_hospitals()

    # Calculate real distance from user and sort by distance
    for h in all_hospitals:
        dist = haversine_distance(lat, lng, h['lat'], h['lng'])
        h['distance'] = f'{dist} km'
        h['distance_val'] = dist

    all_hospitals.sort(key=lambda x: x['distance_val'])

    # Remove internal field
    for h in all_hospitals:
        del h['distance_val']

    return jsonify({'hospitals': all_hospitals})


# ==================== EMERGENCY ACTIONS ====================

@app.route('/api/emergency', methods=['POST'])
def emergency():
    """Handle emergency requests — all actions are logged to MongoDB."""
    data = request.get_json()
    action = data.get('action', '')
    lat = float(data.get('lat', 0))
    lng = float(data.get('lng', 0))

    user = session.get('user', '')
    user_name = session.get('name', 'User')
    contacts = db_get_contacts(user) if user else []

    # Log this emergency action to MongoDB
    db_log_emergency(user, action, lat, lng, len(contacts))

    # Google Maps link for location
    maps_link = f'https://www.google.com/maps?q={lat},{lng}'

    if action == 'send_location':
        # Generate shareable location links for each contact
        sms_links = []
        whatsapp_links = []
        message = f'🚨 EMERGENCY! {user_name} needs help! Location: {maps_link}'

        for c in contacts:
            phone = c['phone'].replace(' ', '').replace('-', '')
            # SMS link
            sms_links.append({
                'name': c['name'],
                'phone': c['phone'],
                'sms_link': f'sms:{phone}?body={message}',
                'whatsapp_link': f'https://wa.me/{phone}?text={message}'
            })

        return jsonify({
            'success': True,
            'message': f'📍 Your location has been prepared for sharing.',
            'details': f'GPS: {lat:.6f}, {lng:.6f}',
            'maps_link': maps_link,
            'emergency_message': message,
            'contact_links': sms_links,
            'contacts_count': len(contacts)
        })

    elif action == 'alert_family':
        if not contacts:
            return jsonify({
                'success': False,
                'message': '⚠️ No family contacts added yet!',
                'details': 'Please add emergency contacts first using the "Manage Contacts" section.',
                'contact_links': [],
                'contacts_count': 0
            })

        message = f'🚨 EMERGENCY ALERT from {user_name}! I need immediate help. My location: {maps_link}'
        alert_links = []

        for c in contacts:
            phone = c['phone'].replace(' ', '').replace('-', '')
            if not phone.startswith('+'):
                phone = '+91' + phone  # Default to India
            alert_links.append({
                'name': c['name'],
                'relationship': c['relationship'],
                'phone': c['phone'],
                'sms_link': f'sms:{phone}?body={message}',
                'whatsapp_link': f'https://wa.me/{phone.replace("+", "")}?text={message}',
                'call_link': f'tel:{phone}'
            })

        return jsonify({
            'success': True,
            'message': f'👨‍👩‍👧‍👦 Emergency alert ready for {len(contacts)} contact(s)!',
            'details': 'Click the links below to send alerts via SMS or WhatsApp.',
            'contact_links': alert_links,
            'contacts_count': len(contacts),
            'emergency_message': message
        })

    elif action == 'call_ambulance':
        return jsonify({
            'success': True,
            'message': '🚑 Call Ambulance Now!',
            'details': 'Tap the number below to call emergency services.',
            'call_links': [
                {'name': '🚑 Ambulance (108)', 'number': '108', 'link': 'tel:108'},
                {'name': '🏥 Emergency (112)', 'number': '112', 'link': 'tel:112'},
                {'name': '🚒 Fire & Rescue (101)', 'number': '101', 'link': 'tel:101'},
            ],
            'maps_link': maps_link
        })

    return jsonify({'error': 'Invalid action'}), 400


# ==================== DB STATUS ENDPOINT ====================

@app.route('/api/db-status')
def db_status():
    """Check MongoDB connection status."""
    if USE_MONGO:
        try:
            info = client.server_info()
            stats = {
                'connected': True,
                'engine': 'MongoDB',
                'version': info.get('version', 'unknown'),
                'database': 'heartguard_db',
                'collections': {
                    'users': db.users.count_documents({}),
                    'contacts': db.contacts.count_documents({}),
                    'hospitals': db.hospitals.count_documents({}),
                    'emergency_logs': db.emergency_logs.count_documents({}),
                }
            }
            return jsonify(stats)
        except Exception as e:
            return jsonify({'connected': False, 'engine': 'MongoDB', 'error': str(e)})
    else:
        return jsonify({
            'connected': False,
            'engine': 'In-Memory (Fallback)',
            'note': 'MongoDB not available. Data is stored in memory and will be lost on restart.',
            'collections': {
                'users': len(MEM_USERS),
                'contacts': sum(len(v) for v in MEM_CONTACTS.values()),
                'hospitals': len(MEM_HOSPITALS),
                'emergency_logs': len(MEM_EMERGENCY_LOGS),
            }
        })

# ==================== DOCTOR APPOINTMENT BOOKING ====================

@app.route('/doctor-appointment')
def doctor_appointment():
    """Doctor appointment booking page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('doctor_appointment.html', name=session.get('name', 'User'))

# ==================== HEALTH SERVICE BOOKING ====================

@app.route('/health-service')
def health_service():
    """Health service booking at home."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('health_service.html', name=session.get('name', 'User'))

# ==================== MEDICINE REMINDER ====================

MEM_MEDICINES = {
    'demo': [
        {'medicine_id': 1, 'username': 'demo', 'medicine_name': 'Aspirin', 'dosage': '75mg', 'time': '08:00 AM', 'date': '2026-03-18'},
        {'medicine_id': 2, 'username': 'demo', 'medicine_name': 'Metoprolol', 'dosage': '50mg', 'time': '08:00 AM', 'date': '2026-03-18'},
        {'medicine_id': 3, 'username': 'demo', 'medicine_name': 'Atorvastatin', 'dosage': '20mg', 'time': '10:00 PM', 'date': '2026-03-18'}
    ],
    'user': [
        {'medicine_id': 1, 'username': 'user', 'medicine_name': 'Clopidogrel', 'dosage': '75mg', 'time': '09:00 AM', 'date': '2026-03-18'},
        {'medicine_id': 2, 'username': 'user', 'medicine_name': 'Ramipril', 'dosage': '5mg', 'time': '09:00 AM', 'date': '2026-03-18'}
    ],
    'doctor1': [
        {'medicine_id': 1, 'username': 'doctor1', 'medicine_name': 'Metformin', 'dosage': '500mg', 'time': '08:00 AM', 'date': '2026-03-18'},
        {'medicine_id': 2, 'username': 'doctor1', 'medicine_name': 'Amlodipine', 'dosage': '5mg', 'time': '08:00 PM', 'date': '2026-03-18'}
    ]
}

@app.route('/medicine-reminder')
def medicine_reminder():
    """Medicine reminder page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    medicines = db_get_medicines(user)
    return render_template('medicine_reminder.html', name=session.get('name', 'User'), medicines=medicines)

def db_get_medicines(username):
    """Get medicines for a user from MongoDB."""
    if USE_MONGO:
        return list(db.medicines.find({'username': username}, {'_id': 0}))
    return MEM_MEDICINES.get(username, [])

@app.route('/api/medicines', methods=['GET', 'POST', 'DELETE'])
def manage_medicines():
    """Manage medicines API."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    user = session['user']
    
    if request.method == 'GET':
        medicines = db_get_medicines(user)
        return jsonify({'medicines': medicines})
    
    if request.method == 'POST':
        data = request.get_json()
        medicine_name = data.get('name', '').strip()
        time = data.get('time', '').strip()
        dosage = data.get('dosage', '').strip()
        
        if not medicine_name or not time:
            return jsonify({'error': 'Medicine name and time are required'}), 400
        
        if USE_MONGO:
            last = db.medicines.find_one({'username': user}, sort=[('medicine_id', -1)])
            next_id = (last['medicine_id'] + 1) if last else 1
            med = {'username': user, 'medicine_id': next_id, 'name': medicine_name, 'time': time, 'dosage': dosage}
            db.medicines.insert_one(med)
            return jsonify({'success': True, 'medicine': med, 'message': f'✅ {medicine_name} reminder set.'})
        else:
            if user not in MEM_MEDICINES:
                MEM_MEDICINES[user] = []
            next_id = len(MEM_MEDICINES[user]) + 1
            med = {'medicine_id': next_id, 'name': medicine_name, 'time': time, 'dosage': dosage}
            MEM_MEDICINES[user].append(med)
            return jsonify({'success': True, 'medicine': med, 'message': f'✅ {medicine_name} reminder set.'})
    
    if request.method == 'DELETE':
        medicine_id = request.args.get('id', type=int)
        if not medicine_id:
            return jsonify({'error': 'Medicine ID required'}), 400
        
        if USE_MONGO:
            db.medicines.delete_one({'username': user, 'medicine_id': medicine_id})
        else:
            if user in MEM_MEDICINES:
                MEM_MEDICINES[user] = [m for m in MEM_MEDICINES[user] if m['medicine_id'] != medicine_id]
        return jsonify({'success': True, 'message': 'Medicine reminder removed.'})

# ==================== HEALTH WORKER MARKETPLACE ====================

@app.route('/health-workers')
def health_workers():
    """Health worker marketplace page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('health_workers.html', name=session.get('name', 'User'))

# ==================== LAB TEST PICKUP ====================

@app.route('/lab-test')
def lab_test():
    """Lab test pickup booking page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('lab_test.html', name=session.get('name', 'User'))

# ==================== HEALTH EQUIPMENT RENTAL ====================

@app.route('/equipment-rental')
def equipment_rental():
    """Health equipment rental page."""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('equipment_rental.html', name=session.get('name', 'User'))

# ==================== BOOKING APIS ====================

@app.route('/api/book-appointment', methods=['POST'])
def book_appointment():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    username = session['user']
    
    # Get next appointment_id
    if USE_MONGO:
        count = db.appointments.count_documents({'username': username})
    else:
        count = len([a for a in MEM_APPOINTMENTS if a['username'] == username])
    appointment_id = count + 1
    
    appointment = {
        'username': username,
        'appointment_id': appointment_id,
        'hospital': data.get('hospital', ''),
        'doctor': data.get('doctor', ''),
        'doctor_username': data.get('doctor_username', ''),
        'specialization': data.get('specialization', ''),
        'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
        'time': data.get('time', ''),
        'patient_name': data.get('patientName', ''),
        'phone': data.get('phone', ''),
        'reason': data.get('reason', 'General Checkup'),
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    }
    
    if USE_MONGO:
        db.appointments.insert_one(appointment)
    else:
        MEM_APPOINTMENTS.append(appointment)
    
    return jsonify({'success': True, 'message': '✅ Appointment booked successfully!', 'appointment_id': appointment_id})


@app.route('/api/appointments', methods=['GET'])
def get_appointments():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['user']
    if USE_MONGO:
        appointments = list(db.appointments.find({'username': username}, {'_id': 0}))
    else:
        appointments = [a for a in MEM_APPOINTMENTS if a['username'] == username]
    return jsonify({'appointments': appointments})

@app.route('/api/book-service', methods=['POST'])
def book_service():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    username = session['user']
    
    # Get next booking_id
    if USE_MONGO:
        count = db.health_services.count_documents({'username': username})
    else:
        count = len([s for s in MEM_HEALTH_SERVICES if s['username'] == username])
    booking_id = count + 1
    
    booking = {
        'username': username,
        'booking_id': booking_id,
        'service_type': 'Health Service',
        'hospital': data.get('hospital', ''),
        'service': data.get('service', ''),
        'patientName': data.get('patientName', ''),
        'phone': data.get('phone', ''),
        'address': data.get('address', ''),
        'nurse_name': data.get('nurseName', ''),
        'nurse_rating': data.get('nurseRating', ''),
        'nurse_approval': data.get('nurseApproval', False),
        'date': data.get('date', ''),
        'time': data.get('time', ''),
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    }
    
    if USE_MONGO:
        db.health_services.insert_one(booking)
    else:
        MEM_HEALTH_SERVICES.append(booking)
    
    return jsonify({'success': True, 'message': '✅ Service booked successfully!', 'booking_id': booking_id})

# Get all bookings (appointments, services, workers, lab tests, equipment)
@app.route('/api/all-bookings', methods=['GET'])
def get_all_bookings():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['user']
    all_bookings = []
    
    if USE_MONGO:
        # Get appointments
        appointments = list(db.appointments.find({'username': username}))
        for appt in appointments:
            appt['_id'] = str(appt['_id'])
            appt['booking_type'] = 'Doctor Appointment'
            all_bookings.append(appt)
        
        # Get health services
        services = list(db.health_services.find({'username': username}))
        for svc in services:
            svc['_id'] = str(svc['_id'])
            svc['booking_type'] = 'Health Service'
            all_bookings.append(svc)
        
        # Get health workers
        workers = list(db.health_workers.find({'username': username}))
        for worker in workers:
            worker['_id'] = str(worker['_id'])
            worker['booking_type'] = 'Health Worker'
            all_bookings.append(worker)
        
        # Get lab tests
        lab_tests = list(db.lab_tests.find({'username': username}))
        for test in lab_tests:
            test['_id'] = str(test['_id'])
            test['booking_type'] = 'Lab Test'
            all_bookings.append(test)
        
        # Get equipment rentals
        rentals = list(db.equipment_rentals.find({'username': username}))
        for rental in rentals:
            rental['_id'] = str(rental['_id'])
            rental['booking_type'] = 'Equipment Rental'
            all_bookings.append(rental)
    else:
        # Get in-memory data
        for appt in [a for a in MEM_APPOINTMENTS if a['username'] == username]:
            all_bookings.append({**appt, 'booking_type': 'Doctor Appointment'})
        for svc in [s for s in MEM_HEALTH_SERVICES if s['username'] == username]:
            all_bookings.append({**svc, 'booking_type': 'Health Service'})
        for w in [w for w in MEM_HEALTH_WORKERS if w['username'] == username]:
            all_bookings.append({**w, 'booking_type': 'Health Worker'})
        for t in [t for t in MEM_LAB_TESTS if t['username'] == username]:
            all_bookings.append({**t, 'booking_type': 'Lab Test'})
        for r in [r for r in MEM_EQUIPMENT_RENTALS if r['username'] == username]:
            all_bookings.append({**r, 'booking_type': 'Equipment Rental'})
    
    # Sort by creation date (newest first)
    all_bookings.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jsonify({'success': True, 'bookings': all_bookings})

# Save payment details
MEM_PAYMENTS = {}

@app.route('/api/save-payment', methods=['POST'])
def save_payment():
    """Save payment details to database."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['user']
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Invalid request data'}), 400
    
    name = data.get('name', '').strip()
    amount = data.get('amount', '').strip()
    upi_id = data.get('upiId', '').strip()
    status = data.get('status', 'completed')
    
    if not name or not amount or not upi_id:
        return jsonify({'error': 'Name, amount and UPI ID are required'}), 400
    
    if USE_MONGO:
        # Get next payment ID
        last = db.payments.find_one({'username': username}, sort=[('payment_id', -1)])
        next_id = (last['payment_id'] + 1) if last else 1
        
        payment = {
            'username': username,
            'payment_id': next_id,
            'name': name,
            'amount': amount,
            'upi_id': upi_id,
            'status': status,
            'created_at': datetime.now().isoformat()
        }
        db.payments.insert_one(payment)
        return jsonify({'success': True, 'payment_id': next_id, 'message': 'Payment saved successfully'})
    else:
        # In-memory fallback
        if username not in MEM_PAYMENTS:
            MEM_PAYMENTS[username] = []
        next_id = len(MEM_PAYMENTS[username]) + 1
        payment = {
            'username': username,
            'payment_id': next_id,
            'name': name,
            'amount': amount,
            'upi_id': upi_id,
            'status': status,
            'created_at': datetime.now().isoformat()
        }
        MEM_PAYMENTS[username].append(payment)
        return jsonify({'success': True, 'payment_id': next_id, 'message': 'Payment saved successfully'})

# Get all payments (for current user)
@app.route('/api/all-payments', methods=['GET'])
def get_all_payments():
    """Get all payment details for current user."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session['user']
    
    if USE_MONGO:
        payments = list(db.payments.find({'username': username}))
        for p in payments:
            p['_id'] = str(p['_id'])
        return jsonify({'success': True, 'payments': payments})
    else:
        return jsonify({'success': True, 'payments': MEM_PAYMENTS.get(username, [])})

# Get ALL payments from ALL users (for admin)
@app.route('/api/admin/all-payments', methods=['GET'])
def get_admin_all_payments():
    """Get all payment details from all users (admin only)."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    # Check if admin
    username = session['user']
    if not username.startswith('admin'):
        return jsonify({'error': 'Admin access required'}), 403
    
    if USE_MONGO:
        payments = list(db.payments.find({}))
        for p in payments:
            p['_id'] = str(p['_id'])
        return jsonify({'success': True, 'payments': payments})
    else:
        all_payments = []
        for user_payments in MEM_PAYMENTS.values():
            all_payments.extend(user_payments)
        return jsonify({'success': True, 'payments': all_payments})

@app.route('/api/book-worker', methods=['POST'])
def book_worker():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    username = session['user']
    
    # Get next booking_id
    if USE_MONGO:
        count = db.health_workers.count_documents({'username': username})
    else:
        count = len([w for w in MEM_HEALTH_WORKERS if w['username'] == username])
    booking_id = count + 1
    
    booking = {
        'username': username,
        'booking_id': booking_id,
        'worker_type': 'Health Worker',
        'worker_name': data.get('workerName', ''),
        'service_type': data.get('serviceType', ''),
        'date': data.get('date', ''),
        'time': data.get('time', ''),
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    }
    
    if USE_MONGO:
        db.health_workers.insert_one(booking)
    else:
        MEM_HEALTH_WORKERS.append(booking)
    return jsonify({'success': True, 'message': '✅ Worker booked successfully!', 'booking_id': booking_id})

@app.route('/api/book-lab-test', methods=['POST'])
def book_lab_test():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    username = session['user']
    
    # Get next booking_id
    if USE_MONGO:
        count = db.lab_tests.count_documents({'username': username})
    else:
        count = len([t for t in MEM_LAB_TESTS if t['username'] == username])
    booking_id = count + 1
    
    booking = {
        'username': username,
        'booking_id': booking_id,
        'test_type': 'Lab Test',
        'test_name': data.get('test', ''),
        'lab_name': data.get('price', ''),
        'patient_name': data.get('patientName', ''),
        'phone': data.get('phone', ''),
        'address': data.get('address', ''),
        'date': data.get('date', ''),
        'time': data.get('time', ''),
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    }
    
    if USE_MONGO:
        db.lab_tests.insert_one(booking)
    else:
        MEM_LAB_TESTS.append(booking)
    return jsonify({'success': True, 'message': '✅ Lab test booked! Agent will visit for sample collection.', 'booking_id': booking_id})

@app.route('/api/rent-equipment', methods=['POST'])
def rent_equipment():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    username = session['user']
    
    # Get next booking_id
    if USE_MONGO:
        count = db.equipment_rentals.count_documents({'username': username})
    else:
        count = len([r for r in MEM_EQUIPMENT_RENTALS if r['username'] == username])
    booking_id = count + 1
    
    booking = {
        'username': username,
        'booking_id': booking_id,
        'equipment_type': 'Equipment Rental',
        'equipment_name': data.get('equipmentName', ''),
        'rental_days': data.get('rentalDays', ''),
        'start_date': data.get('startDate', ''),
        'status': 'booked',
        'created_at': datetime.now().isoformat()
    }
    
    if USE_MONGO:
        db.equipment_rentals.insert_one(booking)
    else:
        MEM_EQUIPMENT_RENTALS.append(booking)
    return jsonify({'success': True, 'message': '✅ Equipment rental request submitted!', 'booking_id': booking_id})


@app.route('/api/update-appointment-status', methods=['POST'])
def update_appointment_status():
    if 'user' not in session or session.get('user') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    appointment_id = data.get('appointment_id')
    username = data.get('username')
    new_status = data.get('status')
    op_type = data.get('type', 'appointment') # Default to appointment
    
    col_map = {
        'appointment': 'appointments',
        'health_service': 'health_services',
        'lab_test': 'lab_tests',
        'equipment_rental': 'equipment_rentals'
    }
    col = col_map.get(op_type, 'appointments')

    if USE_MONGO:
        db[col].update_one(
            {'username': username, 'booking_id': int(appointment_id)} if col != 'appointments' else {'username': username, 'appointment_id': int(appointment_id)},
            {'$set': {'status': new_status}}
        )
    else:
        # Update Memory fallback
        mem_source = MEM_APPOINTMENTS
        if op_type == 'health_service': mem_source = MEM_HEALTH_SERVICES
        elif op_type == 'lab_test': mem_source = MEM_LAB_TESTS
        elif op_type == 'equipment_rental': mem_source = MEM_EQUIPMENT_RENTALS

        # Find specific ID - field name varies based on legacy schema
        id_field = 'appointment_id' if op_type == 'appointment' else 'booking_id'

        for a in mem_source:
            if a['username'] == username and str(a.get(id_field)) == str(appointment_id):
                a['status'] = new_status
                break
                
    return jsonify({'success': True})


@app.route('/api/delete-appointment', methods=['POST'])
def delete_appointment():
    if 'user' not in session or session.get('user') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    appointment_id = data.get('appointment_id')
    username = data.get('username')
    op_type = data.get('type', 'appointment')
    
    col_map = {
        'appointment': 'appointments',
        'health_service': 'health_services',
        'lab_test': 'lab_tests',
        'equipment_rental': 'equipment_rentals'
    }
    col = col_map.get(op_type, 'appointments')
    id_field = 'appointment_id' if op_type == 'appointment' else 'booking_id'

    if USE_MONGO:
        db[col].delete_one({'username': username, id_field: int(appointment_id)})
    else:
        # Update Memory fallback
        if op_type == 'appointment':
            global MEM_APPOINTMENTS
            MEM_APPOINTMENTS = [a for a in MEM_APPOINTMENTS if not (a['username'] == username and str(a.get(id_field)) == str(appointment_id))]
        elif op_type == 'health_service':
            global MEM_HEALTH_SERVICES
            MEM_HEALTH_SERVICES = [s for s in MEM_HEALTH_SERVICES if not (s['username'] == username and str(s.get(id_field)) == str(appointment_id))]
                
    return jsonify({'success': True})


@app.route('/api/update-nurse-status', methods=['POST'])
def update_nurse_status():
    if 'user' not in session or session.get('user') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    nurse_id = data.get('id')
    new_status = data.get('status')
    
    if USE_MONGO:
        db.nurses.update_one({'id': nurse_id}, {'$set': {'status': new_status}})
    else:
        for n in MEM_NURSES:
            if n['id'] == nurse_id:
                n['status'] = new_status
                break
                
    return jsonify({'success': True})

if __name__ == '__main__':
    debug_enabled = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_enabled, host='0.0.0.0', port=port)
