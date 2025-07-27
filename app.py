from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    jsonify
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn
import shutil
from PIL import Image as pImage
import time
import glob

# -------------------------------  App  -----------------------------------------------

app = Flask(__name__)

# ✅ FIX: Absolute SQLite path for Render compatibility
basedir = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, 'database.db')

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "your-secret-key"
db = SQLAlchemy(app)

# ✅ Create DB tables (only if they don't exist)
with app.app_context():
    db.create_all()


# -------------------------------  Database  -----------------------------------------------


# Exampluuu tableuuu jyotiiii

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    videos = db.relationship('Video', backref='user', lazy=True)

class Video(db.Model):
    __tablename__ = "videos"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    is_deepfake = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50))  # For threat categorization
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all();

# -----------------------------   functions   -----------------------------------------------------

# -----------------------------   Routes   -----------------------------------------------------
app.secret_key = "your_secret_key"

# Configuration
UPLOAD_FOLDER = './uploaded_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
app.config['UPLOAD_FOLDER'] = './uploaded_videos'


# Define constants
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((im_size, im_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])




# Define the Model class
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=100, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        # Generate frames from video
        for i, frame in enumerate(self.frame_extract(video_path)):
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            if len(frames) == self.count:
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image
# Utility functions
def split_and_predict(model, video_path, sequence_length=100):
    predictions = []
    confidences = []

    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    num_chunks = total_frames // sequence_length + (1 if total_frames % sequence_length != 0 else 0)
    
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)

    for chunk_idx in range(num_chunks):
        if chunk_idx < len(dataset):
            chunk_frames = dataset[chunk_idx]
            prediction, confidence = predict(model, chunk_frames)

            predictions.append(prediction)
            confidences.append(confidence)
    
    if predictions:
        # Use weighted voting
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        weighted_predictions = []
        for pred, count in zip(unique_predictions, counts):
            avg_conf_for_pred = np.mean([confidences[i] for i in range(len(predictions)) if predictions[i] == pred])
            weighted_predictions.append((pred, count * avg_conf_for_pred))
        
        final_prediction = max(weighted_predictions, key=lambda x: x[1])[0]
        avg_confidence = np.mean(confidences)
    else:
        final_prediction = 0
        avg_confidence = 0.0

    return final_prediction, avg_confidence


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img):
    sm = nn.Softmax(dim=1)
    fmap, logits = model(img.to(device))
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return int(prediction.item()), confidence

# Routes
@app.route('/validate', methods=['GET', 'POST'])
def validate():
    if "user" in session:

        if request.method == 'POST':
            video_file = request.files['video_file']

            if not video_file or not allowed_video_file(video_file.filename):
                return render_template('model_index.html', error="Invalid video file")

            if video_file:
                video_file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
                video_file.save(video_file_path)

                session['file_name'] = video_file_path
                session['sequence_length'] = 100

                return redirect(url_for('predict_page'))

        return render_template('validate.html')
    else:
        return redirect(url_for("login"))

@app.route('/predict')
def predict_page():
    video_file = session.get('file_name')
    sequence_length = session.get('sequence_length', 100)
    model_path = os.path.join('models', 'Model 97 Accuracy 100 Frames FF Data.pt')

    model = Model(2)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model.to(device)
    model.eval()

    final_prediction, avg_confidence = split_and_predict(model, video_file, sequence_length=sequence_length)

    # Determine prediction and category
    if avg_confidence < 40:
        prediction = "UNCERTAIN"
        is_deepfake = None
    elif final_prediction == 0:
        prediction = "REAL" if avg_confidence > 60 else "FAKE"
        is_deepfake = False if avg_confidence > 60 else True
    else:
        prediction = "REAL" if avg_confidence > 60 else "FAKE"
        is_deepfake = True if avg_confidence > 60 else False

    # Save video analysis results
    if "user" in session:
        try:
            new_video = Video(
                user_id=session["user"],
                filename=os.path.basename(video_file),
                is_deepfake=is_deepfake if is_deepfake is not None else False,
                confidence=avg_confidence,
                category='identity_theft'  # You might want to add logic to determine category
            )
            db.session.add(new_video)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"Error saving video: {e}")
            # Handle the error appropriately

    return render_template('predict.html', prediction=prediction, confidence=round(avg_confidence, 1))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/analysis')
def analysis():
    if "user" in session:
        return render_template('analysis.html')
    else:
        return redirect(url_for("login"))

@app.route('/')
def home():
    if "user" in session:
        return render_template('index.html')
    else:
        return redirect(url_for("login"))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/profile')
def profile():
    if "user" in session:
        user=User.query.get(session["user"])
        return render_template('profile.html',user=user)
    else:
        return redirect(url_for("login"))

@app.route("/edit_profile", methods=["POST"])
def edit_profile():
    if "user" in session:
        user = User.query.get(session["user"])

        username = request.form.get("username")
        email = request.form.get("email")

        if email:
            user.email = email
        if username:
            user.username = username

        db.session.commit()

        flash("Profile updated successfully")
        return redirect(url_for("profile"))
    return redirect(url_for("login"))

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup',methods=["POST"])
def signup_post():
    email = request.form.get("email")
    username = request.form.get("username")
    password1 = request.form.get("password1")
    password2 = request.form.get("password2")

    if not username or not email or not password1 or not password2:
            flash("Please enter your details")
            return redirect(url_for("signup"))

    if password1 != password2:
            flash("Passwords do not match")
            return redirect(url_for("signup"))

    user = User.query.filter_by(username=username).first()
    if user:
            flash("Username already exists")
            return redirect(url_for("signup"))

    # Check if email already exists
    existing_email = User.query.filter_by(email=email).first()
    if existing_email:
            flash("Email already exists")
            return redirect(url_for("signup"))

    password_hash = generate_password_hash(password1)

    new_user = User(email=email, username=username, password=password_hash)
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for("login"))


@app.route("/login", methods=["POST"])
def login_post():
    print("request recived")
    username = request.form.get("username")
    password = request.form.get("password")

    if not username or not password:
        flash("Please fill out all fields")
        return redirect(url_for("login"))

    if username == "admin" and password == "admin":
        session["user"] = "admin"
        return redirect(url_for("admin"))

    user = User.query.filter_by(username=username).first()

    if not user:
        flash("Username does not exist")
        return redirect(url_for("login"))

    if not check_password_hash(user.password, password):
        flash("Incorrect password")
        return redirect(url_for("login"))

    session["user"] = user.id
    flash("Login successful")

    return redirect(url_for("profile"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("login")

@app.route('/api/chart-data')
def get_chart_data():
    if "user" not in session:
        return jsonify({
            'totalVideos': 0,
            'deepfakeCount': 0,
            'detectionRates': [],
            'threatCategories': [0, 0, 0, 0]
        })

    user_id = session["user"]
    
    # Get user's videos
    user_videos = Video.query.filter_by(user_id=user_id).all()
    total_videos = len(user_videos)
    deepfake_count = sum(1 for video in user_videos if video.is_deepfake)

    # Calculate detection rates over the last 6 months
    now = datetime.utcnow()
    detection_rates = []
    for i in range(6):
        month_start = now.replace(day=1) - timedelta(days=30 * i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_videos = Video.query.filter(
            Video.user_id == user_id,
            Video.created_at.between(month_start, month_end)
        ).all()
        if month_videos:
            accuracy = sum(1 for v in month_videos if v.confidence > 60) / len(month_videos) * 100
            detection_rates.insert(0, accuracy)
        else:
            detection_rates.insert(0, 0)

    # Get threat categories
    threat_categories = [
        Video.query.filter_by(user_id=user_id, category='identity_theft', is_deepfake=True).count(),
        Video.query.filter_by(user_id=user_id, category='misinformation', is_deepfake=True).count(),
        Video.query.filter_by(user_id=user_id, category='political', is_deepfake=True).count(),
        Video.query.filter_by(user_id=user_id, category='entertainment', is_deepfake=True).count()
    ]

    return jsonify({
        'totalVideos': total_videos,
        'deepfakeCount': deepfake_count,
        'detectionRates': detection_rates,
        'threatCategories': threat_categories
    })

@app.route('/certificate/<int:video_id>')
def show_certificate(video_id):
    if "user" not in session:
        return redirect(url_for("login"))
        
    # Get the video from database
    video = Video.query.get_or_404(video_id)
    
    # Ensure the video belongs to the logged-in user
    if video.user_id != session["user"]:
        flash("Access denied")
        return redirect(url_for("profile"))
    
    # Get the user information
    user = User.query.get(video.user_id)
    
    # Format the timestamp
    formatted_date = video.created_at.strftime("%B %d, %Y at %H:%M:%S UTC")
    
    # Determine verification status and class
    if video.confidence < 40:
        status = "INCONCLUSIVE"
        status_class = "text-yellow-500"
    elif video.is_deepfake:
        status = "DEEPFAKE DETECTED"
        status_class = "text-red-500"
    else:
        status = "AUTHENTIC VIDEO"
        status_class = "text-green-500"
    
    # Generate certificate details
    certificate_data = {
        'video_name': video.filename,
        'status': status,
        'status_class': status_class,
        'confidence': round(video.confidence, 2),
        'analysis_date': formatted_date,
        'user_email': user.email,
        'category': video.category if video.category else 'Not Specified',
        'verification_id': f"VID-{video.id:06d}"  # Format: VID-000001
    }
    
    return render_template('certificate.html', cert=certificate_data)

@app.route('/api/recent-activity')
def get_recent_activity():
    if "user" not in session:
        return jsonify([])

    user_id = session["user"]
    
    # Get user's recent videos (last 5)
    recent_videos = Video.query.filter_by(user_id=user_id)\
        .order_by(Video.created_at.desc())\
        .limit(5)\
        .all()

    activity_data = []
    for video in recent_videos:
        activity_data.append({
            'filename': video.filename,
            'time': video.created_at.strftime('%Y-%m-%d %H:%M'),
            'result': 'Deepfake' if video.is_deepfake else 'Authentic',
            'confidence': round(video.confidence, 1),
            'type': video.category if video.category else 'N/A'
        })

    return jsonify(activity_data)

# if __name__ == '__main__':
#     app.run(debug=True) 
pass
