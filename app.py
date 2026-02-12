from flask import Flask, render_template, request, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
from utils.predictor import predict_cell
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = "hemato_secret_key"

# ---------- DATABASE ----------
BASE_DIR = os.getcwd()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

with app.app_context():
    db.create_all()
# ------------------------------

# ---------- UPLOAD FOLDER ----------
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# -----------------------------------

# ---------- AUTH ----------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # CHECK EMAIL EXISTS
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            return render_template('email_exists.html')  # <-- UPDATED

        # CREATE NEW USER
        user = User(username=username, email=email, password=password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email, password=password).first()

        if user:
            session['user'] = user.username
            return redirect(url_for('home'))
        else:
            return render_template('invalid.html')  # login failed page

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --------------------------------

# ---------- FORCE LOGIN FIRST ----------
@app.route('/')
def index():
    return redirect(url_for('login'))
# --------------------------------------

# ---------- PAGES ----------

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')


@app.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')


@app.route('/contact')
def contact():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

# ---------------------------

# ---------- PREDICT ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    result = predict_cell(path)

    return render_template(
        'result.html',
        prediction=result,
        img_path=f"uploads/{filename}"
    )
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True)
