import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import torch

# Fix for PyTorch 2.6+ weights_only default change
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="Person Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme with neon accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Orbitron:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        text-shadow: 0 0 30px rgba(0, 255, 136, 0.5);
    }
    
    .stats-container {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid #00ff88;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
    }
    
    .stat-value {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #00ff88;
        text-shadow: 0 0 15px rgba(0, 255, 136, 0.8);
    }
    
    .stat-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .detection-card {
        background: rgba(30, 30, 50, 0.9);
        border-left: 3px solid #00d4ff;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .sidebar .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #0a0a0f;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        border: none;
        padding: 0.8rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .sidebar .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(0, 255, 136, 0.4);
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #00ff88, #00d4ff);
    }
    
    div[data-testid="stSidebar"] {
        background: rgba(15, 15, 25, 0.95);
        border-right: 1px solid rgba(0, 255, 136, 0.3);
    }
    
    .coord-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

@st.cache_resource
def load_model(model_path: str = "yolo11n.pt"):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def draw_detections(frame, detections, show_coords=True, box_color=(0, 255, 136), thickness=2):
    """Draw bounding boxes and coordinates on frame"""
    annotated = frame.copy()
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        conf = det['confidence']
        bottom_x, bottom_y = int(det['x']), int(det['y'])
        
        # Draw bounding box with glow effect
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        cv2.rectangle(annotated, (x1-1, y1-1), (x2+1, y2+1), (0, 100, 50), 1)  # Outer glow
        
        # Draw bottom-center point (larger, more visible)
        cv2.circle(annotated, (bottom_x, bottom_y), 8, (0, 212, 255), -1)  # Cyan fill
        cv2.circle(annotated, (bottom_x, bottom_y), 10, (255, 255, 255), 2)  # White border
        
        # Draw label background
        label = f"Person {conf:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), box_color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw coordinates if enabled
        if show_coords:
            coord_text = f"({bottom_x}, {bottom_y})"
            cv2.putText(annotated, coord_text, (bottom_x + 15, bottom_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 212, 255), 2)
    
    return annotated

def detect_persons(model, frame, confidence=0.5):
    """Run YOLO detection on frame"""
    results = model(frame, classes=[0], conf=confidence, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                
                bottom_center_x = (x1 + x2) / 2
                bottom_center_y = y2
                
                detections.append({
                    'x': float(bottom_center_x),
                    'y': float(bottom_center_y),
                    'confidence': conf,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                })
    
    return detections

# Main UI
st.markdown('<h1 class="main-title">🎯 Person Detection</h1>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # Camera selection
    camera_index = st.selectbox(
        "📷 Camera Source",
        options=[0, 1, 2],
        index=0,
        help="Select camera device (0 = default webcam)"
    )
    
    # Confidence threshold
    confidence = st.slider(
        "🎚️ Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for detection"
    )
    
    # Display options
    st.markdown("### 🎨 Display Options")
    show_coords = st.checkbox("Show Coordinates", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    # Box color picker
    box_color_hex = st.color_picker("Box Color", "#00FF88")
    # Convert hex to BGR
    box_color = tuple(int(box_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    
    st.markdown("---")
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶️ Start", use_container_width=True)
    with col2:
        stop_btn = st.button("⏹️ Stop", use_container_width=True)

# Load model
if st.session_state.model is None:
    with st.spinner("🔄 Loading YOLO model..."):
        st.session_state.model = load_model()
        if st.session_state.model:
            st.success("✅ Model loaded successfully!")

# Handle button clicks
if start_btn:
    st.session_state.camera_running = True
if stop_btn:
    st.session_state.camera_running = False

# Main content area
col_video, col_stats = st.columns([3, 1])

with col_video:
    video_placeholder = st.empty()
    
with col_stats:
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    person_count = st.empty()
    fps_display = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 📍 Detections")
    detections_placeholder = st.empty()

# Camera loop
if st.session_state.camera_running and st.session_state.model is not None:
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        st.error(f"❌ Failed to open camera {camera_index}")
        st.session_state.camera_running = False
    else:
        prev_time = datetime.now()
        fps = 0
        
        while st.session_state.camera_running:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ Failed to read frame")
                break
            
            # Calculate FPS
            current_time = datetime.now()
            time_diff = (current_time - prev_time).total_seconds()
            if time_diff > 0:
                fps = 1 / time_diff
            prev_time = current_time
            
            # Run detection
            detections = detect_persons(st.session_state.model, frame, confidence)
            
            # Draw detections
            annotated_frame = draw_detections(frame, detections, show_coords, box_color)
            
            # Add FPS overlay if enabled
            if show_fps:
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Update stats
            person_count.markdown(f"""
                <div style="text-align: center;">
                    <div class="stat-value">{len(detections)}</div>
                    <div class="stat-label">Persons Detected</div>
                </div>
            """, unsafe_allow_html=True)
            
            if show_fps:
                fps_display.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem;">
                        <div class="stat-value" style="font-size: 1.5rem; color: #00d4ff;">{fps:.1f}</div>
                        <div class="stat-label">FPS</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Update detections list
            if detections:
                det_html = ""
                for i, det in enumerate(detections):
                    det_html += f"""
                        <div class="detection-card">
                            <strong>Person {i+1}</strong><br>
                            <span class="coord-display">
                                📍 Bottom: ({int(det['x'])}, {int(det['y'])})<br>
                                📊 Conf: {det['confidence']:.1%}
                            </span>
                        </div>
                    """
                detections_placeholder.markdown(det_html, unsafe_allow_html=True)
            else:
                detections_placeholder.markdown("""
                    <div class="detection-card" style="border-left-color: #666;">
                        <span style="color: #666;">No persons detected</span>
                    </div>
                """, unsafe_allow_html=True)
        
        cap.release()

else:
    # Show placeholder when camera is not running
    video_placeholder.markdown("""
        <div style="
            background: rgba(20, 20, 35, 0.8);
            border: 2px dashed #00ff88;
            border-radius: 12px;
            padding: 4rem;
            text-align: center;
            color: #888;
        ">
            <h2 style="color: #00ff88; font-family: 'Orbitron', monospace;">📷 Camera Ready</h2>
            <p style="font-family: 'JetBrains Mono', monospace;">Click <strong>Start</strong> to begin detection</p>
        </div>
    """, unsafe_allow_html=True)
    
    person_count.markdown("""
        <div style="text-align: center;">
            <div class="stat-value">-</div>
            <div class="stat-label">Persons Detected</div>
        </div>
    """, unsafe_allow_html=True)

