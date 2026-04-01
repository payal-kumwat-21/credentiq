# =========================
# IMPORTS latest file
# =========================
from news_engine import get_news
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import re
from ml.predict import predict_news

# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///credentiq.db'
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("ml/model.pkl", "rb"))
vectorizer = pickle.load(open("ml/vectorizer.pkl", "rb"))

# CLEAN FUNCTION (Unified here)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# =========================
# DATABASE MODELS
# =========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    username = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    is_admin = db.Column(db.Boolean, default=False)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Integer)
    explanation = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# HELPER FUNCTIONS
# =========================
def get_explanation(text, vectorizer, model):
    try:
        feature_names = vectorizer.get_feature_names_out()
        
        if hasattr(model, 'coef_'):
            coefs = model.coef_[0]
        else:
            return "Key signals: " + ", ".join(text.split()[:5])
            
        text_vect = vectorizer.transform([text])
        present_word_indices = text_vect.nonzero()[1]
        
        if len(present_word_indices) == 0:
            return "No strong keywords detected in this text."
            
        word_weight_tuples = []
        for idx in present_word_indices:
            word = feature_names[idx]
            weight = coefs[idx]
            word_weight_tuples.append((word, weight))
            
        word_weight_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
        top_words = [item[0] for item in word_weight_tuples[:5]]
        
        return "Key model indicators detected: " + ", ".join(top_words)
        
    except Exception as e:
        return "Key signals: " + ", ".join(text.split()[:5])

def get_top_words(text, vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    vector = vectorizer.transform([text])
    indices = vector.nonzero()[1]
    words = [feature_names[i] for i in indices[:10]]
    return words

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    news = get_news("general")
    return render_template("index.html", news=news)

@app.route('/category/<cat>')
def category(cat):
    news = get_news(cat)
    return render_template("category.html", news=news, category=cat)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

# ---------- AUTH ----------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not email or not username or not password:
            flash('All fields are required!', 'error')
            return redirect(url_for('signup'))

        existing_user = User.query.filter(
            (User.email == email) |
            (User.username == username)
        ).first()
        
        if existing_user:
            flash('This email or username is already registered!', 'error')
            return redirect(url_for('signup'))
            
        hashed_password = generate_password_hash(password)
        new_user = User(
            email=email,
            username=username,
            password=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password. Please try again.', 'error')
        return redirect(url_for('login'))
    return render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

# ---------- ANALYTICS ----------
@app.route('/analytics')
@login_required
def analytics():
    history = Analysis.query.filter_by(user_id=current_user.id).all()
    fake_count = len([h for h in history if "Fake" in h.result])
    real_count = len([h for h in history if "Real" in h.result])
    return render_template("analytics.html", fake_count=fake_count, real_count=real_count)

# ---------- QUICK ANALYZE ----------
@app.route('/quick_analyze', methods=['POST'])
def quick_analyze():
    try:
        text = request.form.get('news')
        if not text or text.strip() == "":
            return "Error: No news data received"
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        prob = model.predict_proba(vect).max()
        result = "Fake News" if pred == 0 else "Real News"
        confidence = int(prob * 100)
        explanation = get_explanation(text, vectorizer, model)
        top_words = get_top_words(cleaned, vectorizer, model)
        return render_template(
            "quick_result.html",
            text=text,
            result=result,
            confidence=confidence,
            explanation=explanation,
            words=top_words
        )
    except Exception as e:
        return f"ERROR: {str(e)}"

# ---------- DASHBOARD ----------
@app.route('/dashboard')
@login_required
def dashboard():
    history = Analysis.query.filter_by(user_id=current_user.id).all()
    return render_template("dashboard.html",
        history=history,
        total=len(history),
        fake_count=len([h for h in history if "Fake" in h.result]),
        real_count=len([h for h in history if "Real" in h.result])
    )

# ---------- ANALYZE ----------
@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    text = request.form['content']
    vect = vectorizer.transform([text])
    pred = model.predict(vect)[0]
    prob = np.max(model.predict_proba(vect))
    result = "Fake News" if pred == 0 else "Real News"
    confidence = int(prob * 100)
    
    # Corrected the function call here:
    explanation = get_explanation(text, vectorizer, model)
    
    record = Analysis(
        content=text,
        result=result,
        confidence=confidence,
        explanation=explanation,
        user_id=current_user.id
    )
    db.session.add(record)
    db.session.commit()
    return redirect('/dashboard')

# ---------- HISTORY ----------
@app.route('/history')
@login_required
def history():
    user_history = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.id.desc()).all()
    return render_template("history.html", history=user_history)

# =========================
# RUN
# =========================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(port=5000)
