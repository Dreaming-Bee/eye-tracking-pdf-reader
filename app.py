# app.py
import os
import cv2
import numpy as np
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PDF_FOLDER'] = 'static/pdfs/'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_FOLDER'], exist_ok=True)

# Global variables for eye tracking
eye_tracking_active = False
current_page = 0
total_pages = 0
current_pdf = None
scroll_position = 0
last_scroll_time = 0
scroll_cooldown = 0.5  # seconds between scrolls

# Eye tracking setup with MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def convert_pdf_to_images(pdf_path, output_dir):
    """Convert PDF pages to images and save them"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    for page_number in range(total_pages):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        output_path = os.path.join(output_dir, f"page_{page_number + 1}.png")
        pix.save(output_path)
    
    return total_pages

def get_eye_direction(landmarks, frame_shape):
    """Determine eye gaze direction based on eye landmarks"""
    # Get eye landmarks
    # Left eye: landmarks 362, 385, 387, 263, 373, 380
    # Right eye: landmarks 33, 160, 158, 133, 153, 144
    
    left_eye_landmarks = [landmarks[362], landmarks[385], landmarks[387], 
                          landmarks[263], landmarks[373], landmarks[380]]
    right_eye_landmarks = [landmarks[33], landmarks[160], landmarks[158], 
                          landmarks[133], landmarks[153], landmarks[144]]
    
    # Calculate eye centers
    left_eye_center = np.mean(np.array([(l.x * frame_shape[1], l.y * frame_shape[0]) 
                                       for l in left_eye_landmarks]), axis=0)
    right_eye_center = np.mean(np.array([(l.x * frame_shape[1], l.y * frame_shape[0]) 
                                        for l in right_eye_landmarks]), axis=0)
    
    # Calculate iris positions (landmarks 473, 468 are iris centers in MediaPipe Face Mesh)
    left_iris_center = np.array([landmarks[473].x * frame_shape[1], landmarks[473].y * frame_shape[0]])
    right_iris_center = np.array([landmarks[468].x * frame_shape[1], landmarks[468].y * frame_shape[0]])
    
    # Compare iris positions to eye centers to determine gaze direction
    left_eye_direction = left_iris_center - left_eye_center
    right_eye_direction = right_iris_center - right_eye_center
    
    # Average the directions from both eyes
    avg_vertical_direction = (left_eye_direction[1] + right_eye_direction[1]) / 2
    
    # Determine if looking up or down
    threshold = 1.0  # Adjust as needed
    if avg_vertical_direction < -threshold:
        return "up"
    elif avg_vertical_direction > threshold:
        return "down"
    else:
        return "center"

def eye_tracking_thread():
    """Thread function for eye tracking"""
    global eye_tracking_active, current_page, total_pages, scroll_position, last_scroll_time
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam")
        return
    
    while eye_tracking_active:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            gaze_direction = get_eye_direction(landmarks, frame.shape)
            
            # Handle scrolling based on gaze direction
            current_time = time.time()
            if current_time - last_scroll_time > scroll_cooldown:
                if gaze_direction == "down" and scroll_position < 100:
                    scroll_position += 10
                    last_scroll_time = current_time
                elif gaze_direction == "up" and scroll_position > 0:
                    scroll_position -= 10
                    last_scroll_time = current_time
        
        time.sleep(0.01)  # Reduce CPU usage
    
    cap.release()

@app.route('/')
def index():
    # Get list of uploaded PDFs
    pdfs = []
    if os.path.exists(app.config['PDF_FOLDER']):
        pdfs = [f for f in os.listdir(app.config['PDF_FOLDER']) if f.endswith('.pdf')]
    
    return render_template('index.html', pdfs=pdfs)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Move to PDF folder for viewing
        pdf_path = os.path.join(app.config['PDF_FOLDER'], filename)
        os.rename(file_path, pdf_path)
        
        # Convert PDF to images
        total_pages = convert_pdf_to_images(pdf_path, app.config['PDF_FOLDER'])
        
        return redirect(url_for('index'))
    
    return redirect(request.url)

@app.route('/view/<path:pdf_name>')
def view_pdf(pdf_name):
    global current_pdf, total_pages, current_page, eye_tracking_active, scroll_position
    
    pdf_path = os.path.join(app.config['PDF_FOLDER'], pdf_name)
    if not os.path.exists(pdf_path):
        return redirect(url_for('index'))
    
    # Get PDF information
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    current_page = 0
    current_pdf = pdf_name
    scroll_position = 0
    
    return render_template('viewer.html', pdf_name=pdf_name, total_pages=total_pages)

@app.route('/start_eye_tracking', methods=['POST'])
def start_eye_tracking():
    global eye_tracking_active
    
    if not eye_tracking_active:
        eye_tracking_active = True
        threading.Thread(target=eye_tracking_thread).start()
    
    return jsonify({"status": "started"})

@app.route('/stop_eye_tracking', methods=['POST'])
def stop_eye_tracking():
    global eye_tracking_active
    
    eye_tracking_active = False
    
    return jsonify({"status": "stopped"})

@app.route('/get_scroll_position')
def get_scroll_position():
    global scroll_position
    
    return jsonify({"scroll_position": scroll_position})

@app.route('/static/pdfs/<path:filename>')
def serve_pdf(filename):
    return send_from_directory(app.config['PDF_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)