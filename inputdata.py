import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
from collections import deque
import copy
import mediapipe as mp
import csv
from app import calc_landmark_list, pre_process_landmark

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        print(f"Logged data for label {number}: {landmark_list}")
    if mode == 2:
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def write_label_to_csv(label):
    label_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    with open(label_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label])
    
    with open(label_path, 'r') as f:
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

# Streamlit App
st.title("Hand Gesture Data Logger")

# Sidebar for input
st.sidebar.title("Input Label")
label = st.sidebar.text_input("Enter a label for the gesture:", "")
start_logging = st.sidebar.button("Start Logging")

# Webcam feed
st.header("Webcam Feed")
video_placeholder = st.empty()

# State for holding the logging status
is_logging = st.session_state.get("is_logging", False)

# Start/Stop logging button
if "is_logging" not in st.session_state:
    st.session_state.is_logging = False

toggle_logging = st.button("Hold to Log")

if toggle_logging:
    st.session_state.is_logging = True
else:
    st.session_state.is_logging = False
###################
if start_logging and label:
    number= write_label_to_csv(label)
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access the webcam.")
            break

        # Flip the frame for a mirror effect
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        results = hands.process(frame_rgb)

        # Process landmarks if a hand is detected
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Preprocess landmarks
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Log data to CSV
                if st.session_state.is_logging:
                    logging_csv(number, 1, pre_processed_landmark_list, [])
                    st.write(f"Logged data for label: {label}")

                # Draw landmarks on the frame
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )


        # Convert the frame to RGB for Streamlit
        frame_rgb = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Display the frame in Streamlit
        video_placeholder.image(frame_pil, caption="Webcam Feed", use_container_width=True)

        # Stop logging on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    st.write("Logging stopped.")
else:
    st.write("Enter a label and click 'Start Logging' to begin.")
