import torch
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import collections
import tempfile
import os

# Set page layout and aesthetics
st.set_page_config(
    page_title="YOLO 2D Detection & Tracking Showcase",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subheader-text {
            font-size: 1.1rem;
            color: #6c757d;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border-left: 5px solid #FF4B4B;
        }
        .dark .metric-card {
            background-color: #1e2430;
            border-left: 5px solid #FF4B4B;
        }
        .info-label {
            font-weight: bold;
            color: #FF4B4B;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("<h1 style='text-align: center;'>⚙️ Dashboard Configuration</h1>", unsafe_allow_html=True)
st.sidebar.write("Configure YOLO Model & Pipeline settings below:")

# Task Selection
task = st.sidebar.selectbox(
    "Choose AI Task",
    ["Object Detection", "Object Tracking", "Image Classification"],
    index=0
)

# Model Version Selection
model_family = st.sidebar.radio(
    "Model Family",
    ["YOLO11 (Latest)", "YOLOv8"],
    horizontal=True
)

# Model Scale Selection
model_scale = st.sidebar.select_slider(
    "Model Size (Complexity)",
    options=["nano", "small", "medium", "large", "xlarge"],
    value="nano",
    help="Nano is faster and lighter; Large/XLarge are more accurate but compute-heavy."
)

# Map human-readable model sizes to Ultralytics model naming
scale_map = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
suffix = scale_map[model_scale]
is_cls = task == "Image Classification"

if model_family.startswith("YOLO11"):
    model_name = f"yolo11{suffix}-cls.pt" if is_cls else f"yolo11{suffix}.pt"
else:
    model_name = f"yolov8{suffix}-cls.pt" if is_cls else f"yolov8{suffix}.pt"

# Settings sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence score for detections to be shown."
)

iou_threshold = 0.45
if task != "Image Classification":
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Maximum Suppression."
    )

# Tracking visual option
draw_trails = False
if task == "Object Tracking":
    draw_trails = st.sidebar.checkbox(
        "Draw Track History Trails",
        value=True,
        help="Display colorful historical path lines behind moving tracked objects."
    )

# Cache YOLO models to prevent reloading on page updates
@st.cache_resource
def load_yolo_model(name):
    try:
        return YOLO(name)
    except Exception as e:
        st.error(f"Error loading model '{name}': {e}")
        return None

# Load the chosen model
with st.sidebar:
    st.markdown("---")
    model_load_status = st.empty()
    model_load_status.info(f"⏳ Loading model `{model_name}`...")
    
model = load_yolo_model(model_name)

if model:
    st.sidebar.success(f"✓ Model `{model_name}` active!")
else:
    st.sidebar.error("✗ Failed to load selected model.")
    st.stop()

# ----------------- MAIN PANEL -----------------
st.markdown("<div class='main-header'>Pretrained YOLO 2D Detection & Tracking Showcase</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-text'>Analyze images, videos, or camera feeds in real-time using industry-standard models.</div>", unsafe_allow_html=True)

# Select Input Source
input_type = st.radio(
    "Select Input Source",
    ["Upload Files", "Sample Media", "Use Webcam (Live)"],
    horizontal=True
)

source_file = None
is_sample = False
use_webcam = False
camera_index = 0

sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

if input_type == "Upload Files":
    file_types = ["png", "jpg", "jpeg"]
    if task != "Image Classification":
        file_types.extend(["mp4", "avi", "mov", "mkv"])
        
    source_file = st.file_uploader(
        "Choose an image or video file...",
        type=file_types
    )
elif input_type == "Sample Media":
    is_sample = True
    sample_files = []
    
    # Check what files are available in data/ directory
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if not f.startswith(".")]
        
    if not sample_files:
        st.warning("⚠️ No sample media found. Run `python download_sample.py` to fetch sample images and videos.")
        st.info("Falling back to standard internet test urls...")
        # Fallbacks
        if task == "Image Classification" or task == "Object Detection":
            selected_sample = "https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/bus.jpg"
        else:
            selected_sample = "https://github.com/ultralytics/assets/releases/download/v8.2.0/california.mp4"
    else:
        selected_sample_name = st.selectbox(
            "Choose a sample file to run:",
            sample_files
        )
        selected_sample = os.path.join(sample_dir, selected_sample_name)
else:
    use_webcam = True
    camera_index = st.sidebar.number_input(
        "Camera Device Index",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Select camera device index. Default is 0 (built-in webcam). Try 1 or 2 if your Pixel 8 is connected as a USB webcam."
    )

# Handle execution
if source_file is not None or is_sample or use_webcam:
    
    # ----------------- PROCESS IMAGE -----------------
    # Helper to check if file/sample is image
    is_image = False
    if source_file:
        is_image = source_file.name.split('.')[-1].lower() in ["png", "jpg", "jpeg"]
    elif is_sample:
        is_image = selected_sample.split('.')[-1].lower() in ["png", "jpg", "jpeg"] or "bus.jpg" in selected_sample
        
    if is_image and not use_webcam:
        # Load image
        if source_file:
            image = Image.open(source_file)
        else:
            if selected_sample.startswith("http"):
                # Download on the fly if web URL fallback is used
                import requests
                from io import BytesIO
                response = requests.get(selected_sample)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(selected_sample)
                
        # Run inference
        st.write("### Processing Image...")
        img_array = np.array(image)
        
        # Run model
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            iou=iou_threshold if not is_cls else None
        )
        
        res = results[0]
        
        # Display side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", width='stretch')
            
        with col2:
            annotated_img = res.plot()
            # Convert BGR back to RGB for display
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            st.image(annotated_img_rgb, caption="YOLO Result", use_container_width=True)
            
        # Display tasks-specific details
        st.markdown("---")
        st.write("### Inference Analysis")
        
        if is_cls:
            if res.probs is not None:
                top5_indices = res.probs.top5
                top5_confs = [float(c) for c in res.probs.top5conf]
                
                # Layout top predicted class with large text
                top1_idx = res.probs.top1
                top1_name = res.names[top1_idx]
                top1_conf = float(res.probs.top1conf)
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <p style='margin:0; font-size: 0.9rem; color: #888;'>TOP PREDICTED CLASS</p>
                            <h2 style='margin:5px 0 0 0;'>{top1_name}</h2>
                            <p style='margin:5px 0 0 0; font-size: 1.5rem; font-weight:bold; color:#FF4B4B;'>{top1_conf*100:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c2:
                    # Bar chart of top 5
                    chart_data = {
                        "Class": [res.names[idx] for idx in top5_indices],
                        "Confidence": top5_confs
                    }
                    st.bar_chart(data=chart_data, x="Class", y="Confidence", color="#FF4B4B")
        else:
            # Detection or Tracking
            boxes = res.boxes
            if boxes is not None:
                num_objects = len(boxes)
                st.markdown(f"**Total Objects Detected:** `{num_objects}`")
                
                if num_objects > 0:
                    # Table details
                    obj_details = []
                    class_counts = {}
                    
                    for i in range(num_objects):
                        cls_id = int(boxes.cls[i])
                        name = res.names[cls_id]
                        conf = float(boxes.conf[i])
                        box_coords = [round(c, 1) for c in boxes.xyxy[i].tolist()]
                        
                        class_counts[name] = class_counts.get(name, 0) + 1
                        
                        obj_details.append({
                            "Object #": i + 1,
                            "Class": name,
                            "Confidence": f"{conf:.2%}",
                            "Bounding Box [xmin, ymin, xmax, ymax]": str(box_coords)
                        })
                    
                    # Columns display
                    st.dataframe(obj_details, use_container_width=True)
                    
                    # Mini bar chart of counts
                    st.write("#### Class Counts")
                    st.bar_chart(class_counts)
                else:
                    st.info("No objects detected above the confidence threshold.")
                    
    # ----------------- PROCESS VIDEO / WEBCAM -----------------
    else:
        # Video source resolution
        tfile = None
        cap = None
        
        if use_webcam:
            st.write("### Processing Live Webcam Feed...")
            st.info("Press the 'Stop Live Feed' button below to terminate.")
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error("Error: Webcam could not be accessed.")
                st.stop()
        else:
            st.write("### Processing Video...")
            # Streamlit needs local file path for cv2.VideoCapture
            if source_file:
                # Save uploaded file to temp file to read with cv2
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(source_file.read())
                video_path = tfile.name
                tfile.close() # Close to allow opencv to open it on Windows
            else:
                video_path = selected_sample
                
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"Error: Could not open video file {video_path}")
                if tfile:
                    os.unlink(tfile.name)
                st.stop()
                
        # Layout holders
        stop_clicked = st.button("Stop Processing" if not use_webcam else "Stop Live Feed")
        
        frame_placeholder = st.empty()
        
        # Bottom metrics panel
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            fps_holder = st.empty()
        with metric_col2:
            count_holder = st.empty()
        with metric_col3:
            track_holder = st.empty()
            
        summary_holder = st.empty()
        
        # Track history initialization
        track_history = collections.defaultdict(lambda: [])
        
        # Loop over video frames
        import time
        prev_time = time.time()
        
        frame_idx = 0
        
        while cap.isOpened() and not stop_clicked:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Perform prediction/tracking
            if task == "Object Tracking":
                results = model.track(
                    source=frame,
                    persist=True,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
            else:
                # Normal detection or classification on video frame
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    iou=iou_threshold if not is_cls else None,
                    verbose=False
                )
                
            res = results[0]
            annotated_frame = res.plot()
            
            # Custom line tracking draw
            if task == "Object Tracking" and draw_trails:
                boxes = res.boxes
                if boxes is not None and boxes.id is not None:
                    xywhs = boxes.xywh.cpu().numpy()
                    ids = boxes.id.int().cpu().tolist()
                    
                    for box, track_id in zip(xywhs, ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:  # Retain last 30 center locations
                            track.pop(0)
                            
                        # Draw tracking paths on frame
                        if len(track) > 1:
                            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                            # Draw a multi-colored/custom colored trail line
                            # Color is deterministically selected based on track id
                            color = (
                                int((track_id * 107) % 255),
                                int((track_id * 73) % 255),
                                int((track_id * 163) % 255)
                            )
                            cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=3)
                            
            # Convert BGR frame to RGB for streamlit
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update frame visual
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time
            
            # Update metric cards
            fps_holder.markdown(f"""
                <div class='metric-card'>
                    <p style='margin:0; font-size: 0.85rem; color: #888;'>PROCESSING SPEED</p>
                    <h3 style='margin:5px 0 0 0;'>{fps:.1f} FPS</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if is_cls:
                if res.probs is not None:
                    top1_name = res.names[res.probs.top1]
                    top1_conf = float(res.probs.top1conf)
                    count_holder.markdown(f"""
                        <div class='metric-card'>
                            <p style='margin:0; font-size: 0.85rem; color: #888;'>TOP CLASSIFICATION</p>
                            <h3 style='margin:5px 0 0 0;'>{top1_name} ({top1_conf:.1%})</h3>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                boxes = res.boxes
                num_detected = len(boxes) if boxes is not None else 0
                count_holder.markdown(f"""
                    <div class='metric-card'>
                        <p style='margin:0; font-size: 0.85rem; color: #888;'>OBJECTS IN FRAME</p>
                        <h3 style='margin:5px 0 0 0;'>{num_detected}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Unique track ids seen
                if task == "Object Tracking" and boxes is not None and boxes.id is not None:
                    unique_tracks = len(track_history.keys())
                    track_holder.markdown(f"""
                        <div class='metric-card'>
                            <p style='margin:0; font-size: 0.85rem; color: #888;'>TOTAL TRACKS SEEN</p>
                            <h3 style='margin:5px 0 0 0;'>{unique_tracks}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    track_holder.markdown(f"""
                        <div class='metric-card'>
                            <p style='margin:0; font-size: 0.85rem; color: #888;'>TRACKING METRICS</p>
                            <h3 style='margin:5px 0 0 0; color:gray;'>N/A (Tracking off)</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
            # Check if stop clicked inside loop
            # streamlit evaluates this on button press, but within loop we need to check stream state
            # If the user presses the button, streamlit rerun will naturally terminate
            
        cap.release()
        st.success("✅ Finished processing video.")
        
        # Clean up temporary file
        if tfile and os.path.exists(tfile.name):
            try:
                os.unlink(tfile.name)
            except Exception:
                pass
