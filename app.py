from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import uuid
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import Model
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///project.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(120), nullable=False)
    videos = db.relationship('Video', backref='user', lazy=True)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    result = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Model Setup - Force CPU for production deployment
device = torch.device('cpu')
model = Model(num_classes=2)
model_path = os.path.join('models', 'Model 97 Accuracy 100 Frames FF Data.pt')

try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
except Exception as e:
    print(f"Error loading model: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_video(video_path, sequence_length=20):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return False, 0.0
        
        # Smart frame sampling
        interval = total_frames / sequence_length
        frame_indices = [int(i * interval) for i in range(min(sequence_length, total_frames))]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor_img = transform(Image.fromarray(frame_rgb))
                frames.append(tensor_img)
        cap.release()
        
        if not frames:
            return False, 0.0
        
        video_tensor = torch.stack(frames).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(video_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probabilities[0][1].item()
            is_deepfake = fake_prob > 0.5
            confidence = fake_prob * 100 if is_deepfake else (1.0 - fake_prob) * 100
        
        # Cleanup
        del video_tensor, outputs, probabilities
        
        return is_deepfake, confidence
    except Exception as e:
        print(f"Error in prediction: {e}")
        return False, 0.0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password1')
        password2 = request.form.get('password2')

        if password != password2:
            flash('Passwords do not match')
            return redirect(url_for('signup'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('signup'))

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/validate', methods=['GET', 'POST'])
@login_required
def validate():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            flash('No video file uploaded')
            return redirect(request.url)
            
        video_file = request.files['video_file']
        if video_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if video_file:
            filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(video_file.filename))[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video_file.save(filepath)

            try:
                is_deepfake, confidence = predict_video(filepath)
                result_str = 'Deepfake' if is_deepfake else 'Authentic'
                
                video = Video(
                    filename=filename,
                    result=result_str,
                    confidence=confidence,
                    user_id=current_user.id
                )
                db.session.add(video)
                db.session.commit()

                return render_template('predict.html', 
                                    prediction=result_str,
                                    confidence=round(confidence, 2),
                                    unique_hash_id=filename)
            except Exception as e:
                print(f"Error processing video: {e}")
                flash('Error processing video')
                return redirect(request.url)

    return render_template('validate.html')

@app.route('/api/stats')
def get_stats():
    total_videos = Video.query.count()
    deepfake_count = Video.query.filter_by(result='Deepfake').count()
    authentic_count = Video.query.filter_by(result='Authentic').count()
    
    return jsonify({
        'total_videos': total_videos,
        'deepfake_count': deepfake_count,
        'authentic_count': authentic_count
    })

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
    
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))

