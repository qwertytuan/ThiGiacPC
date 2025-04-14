import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from collections import deque
import copy
import mediapipe as mp
import csv
from app import calc_landmark_list, pre_process_landmark,pre_process_point_history
from model import KeyPointClassifier
from model import PointHistoryClassifier



def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="",encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        print(f"Logged data for label {number}: {landmark_list}")
    if mode == 2:
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="",encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def read_labels_from_csv(label_path):
    labels = []
    try:
        with open(label_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            labels = [row[0] for row in reader]
    except FileNotFoundError:
        st.error(f"File not found: {label_path}")
    return labels



def write_label_to_csv(label,mode):
    if mode == 1:
        label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    elif mode == 2:
        label_path = 'model/point_history_classifier/point_history_classifier_label.csv'
    with open(label_path, 'a', newline="",encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([label])
    
    with open(label_path, 'r',encoding='utf-8-sig') as f:
        num_rows = sum(1 for _ in f)
    return num_rows - 1  # Return the index of the label (0-based)

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

##############
keypoint_classifier = KeyPointClassifier()

point_history_classifier = PointHistoryClassifier()
    # Read labels ###########################################################
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
with open('model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

point_history = deque(maxlen=16)
##############
st.title("Hand Gesture Data Logger")
# Sidebar for displaying labels
st.sidebar.title("Existing Labels")
st.sidebar.write("View all existing labels for gestures.")

# Display Keypoint Labels
st.sidebar.subheader("Keypoint Labels")
keypoint_labels = read_labels_from_csv('model/keypoint_classifier/keypoint_classifier_label.csv')
st.sidebar.write("Point Labels:")
point_labels = read_labels_from_csv('model/point_history_classifier/point_history_classifier_label.csv')
if keypoint_labels:
    for i, label in enumerate(keypoint_labels):
        st.sidebar.write(f"{i}: {label}")


# Sidebar for input
st.sidebar.title("Gesture Logging")
st.sidebar.write("Log hand gestures for machine learning model training.")
st.sidebar.write("Ensure your webcam is enabled and working.")
st.sidebar.write("Press 'Start Logging' to begin capturing gestures.")
st.sidebar.write("Press 'Capture' to log the gesture data.")
st.sidebar.write("Note: The label will be saved to a CSV file.")
st.sidebar.title("Input Label")
label = st.sidebar.text_input("Enter a label for the gesture:", "")

# Add a select box for mode selection
mode = st.selectbox("Select a mode:", ["Keypoint", "Point History"])
# Start logging button
start_logging = st.sidebar.button("Start Logging")

# Webcam feed
st.header("Webcam Feed")
video_placeholder = st.empty()

if start_logging and label:
    if mode == "Keypoint":
        mode2 = 1
        st.write("Logging keypoint data...")
    elif mode == "Point History":
        mode2 = 2
        st.write("Logging point history data...")
    st.write(f"Label: {label}")
    number = write_label_to_csv(label,mode2)
    cap = cv.VideoCapture(0)  # type: ignore
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Flip the frame for a mirror effect
        frame = cv.flip(frame, 1)  # type: ignore
        debug_image = copy.deepcopy(frame)

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # type: ignore

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)
        
        # Coordinate history #################################################################
        # Process landmarks if a hand is detected
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                
                
                # Log data to CSV based on the selected mode
                if mode2 == 1:
                    logging_csv(number, 1, pre_processed_landmark_list, pre_processed_point_history_list)
                    print(pre_processed_landmark_list)
                    print(point_history)
                elif mode2 == 2:
                    logging_csv(number, 2, pre_processed_landmark_list, pre_processed_point_history_list)
                    print(pre_processed_point_history_list)
      

                # Draw landmarks on the frame
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
        else:
            point_history.append([0, 0])     

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)  # type: ignore
        frame_pil = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit
        video_placeholder.image(frame_pil, caption="Webcam Feed", use_container_width=True)

    cap.release()
    st.write("Logging stopped.")
else:
    st.write("Enter a label and click 'Start Logging' to begin.")