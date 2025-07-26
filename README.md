# Deepfake Detection

This project is a web application for detecting deepfake videos using a PyTorch-based model and a Flask backend. Users can upload videos, receive predictions, and view analysis and certificates for their uploads.

## Features
- User authentication (signup, login, profile management)
- Video upload and deepfake detection
- Confidence scoring and threat categorization
- User dashboard with analytics and recent activity
- Certificate generation for video authenticity

## Tech Stack
- **Backend:** Flask, Flask-SQLAlchemy
- **Frontend:** HTML, CSS, JavaScript (Jinja2 templates)
- **ML/DL:** PyTorch, Torchvision, OpenCV
- **Database:** SQLite

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/ThakurSaiPrakash12/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Create a virtual environment and activate it:**
   ```sh
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirement.txt
   ```

4. **Run the application:**
   ```sh
   python app.py
   ```

5. **Access the app:**
   Open your browser and go to `http://127.0.0.1:5000/`

## Project Structure
```
Deepfake Detection/
├── app.py
├── models/
│   └── Model 97 Accuracy 100 Frames FF Data.pt
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/
├── uploaded_videos/
├── instance/
│   └── database.db
├── requirement.txt
└── README.md
```

## Notes
- The pre-trained model file should be placed in the `models/` directory.
- Uploaded videos are stored in the `uploaded_videos/` directory.
- The database is stored in `instance/database.db`.

## License
This project is for educational purposes.

---

Feel free to contribute or raise issues! 