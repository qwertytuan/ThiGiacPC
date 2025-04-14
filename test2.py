import csv
import copy
from collections import Counter, deque
import subprocess
import cv2 as cv
import mediapipe as mp
import streamlit as st
from PIL import Image
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import app as ap
import time
# Function to execute a shell command
def execute_command(command, delay=2):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        st.write(f"Command executed successfully: {command}")
        st.write(f"Output: {result.stdout}")
        time.sleep(delay)  # Add a delay after executing the command
    except subprocess.CalledProcessError as e:
        st.error(f"Error while executing command: {e.stderr}")
        time.sleep(delay)  # Add a delay even if the command fails


# Function to load hand sign labels
def load_hand_sign_labels(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            labels = [row[0] for row in reader]
        return labels
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return []

# Function to save hand sign-to-command mappings
def save_hand_sign_commands(mapping, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        for hand_sign, command in mapping.items():
            writer.writerow([hand_sign, command])

# Function to load hand sign-to-command mappings
def load_hand_sign_commands(file_path):
    mapping = {}
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    mapping[row[0]] = row[1]
    except FileNotFoundError:
        pass  # If the file doesn't exist, return an empty mapping
    return mapping

# Main Streamlit app
def main():
    st.title("Hand Sign Command Control")

    # Load hand sign labels
    hand_sign_labels = load_hand_sign_labels('model/keypoint_classifier/keypoint_classifier_label.csv')

    # Load existing hand sign-to-command mappings
    command_mapping = load_hand_sign_commands('hand_sign_commands.csv')

    # Sidebar for assigning commands to hand signs
    st.sidebar.title("Assign Commands to Hand Signs")
    selected_hand_sign = st.sidebar.selectbox("Select a hand sign:", hand_sign_labels)
    command = st.sidebar.text_input("Enter the command for this hand sign:")
    if st.sidebar.button("Assign Command"):
        if selected_hand_sign and command:
            command_mapping[selected_hand_sign] = command
            save_hand_sign_commands(command_mapping, 'hand_sign_commands.csv')
            st.sidebar.success(f"Assigned command '{command}' to hand sign '{selected_hand_sign}'.")
        else:
            st.sidebar.error("Please select a hand sign and enter a command.")

    # Display existing mappings
    st.sidebar.subheader("Existing Hand Sign Commands")
    if command_mapping:
        for hand_sign, command in command_mapping.items():
            st.sidebar.write(f"**{hand_sign}**: {command}")
    else:
        st.sidebar.write("No commands assigned yet.")

    # Webcam feed and hand sign detection
    st.header("Webcam Feed")
    cap_device = st.selectbox("Select Camera", ["Web Cam", "External Cam"])
    cap_device = 0 if cap_device == "Web Cam" else 2
    cap_width = st.slider("Camera Width", min_value=10, max_value=1920, value=1280, step=10)
    cap_height = st.slider("Camera Height", min_value=10, max_value=1080, value=720, step=10)
    start_webcam = st.button("Start Webcam")

    if start_webcam:
        st.write("Webcam started.")
        video_placeholder = st.empty()

        # Initialize Mediapipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # Initialize KeyPoint Classifier
        keypoint_classifier = KeyPointClassifier()

        # Camera preparation
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        
        last_command_time = 0  
        command_delay = 2 
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
                    landmark_list = ap.calc_landmark_list(debug_image, hand_landmarks)

                    # Preprocess landmarks
                    pre_processed_landmark_list = ap.pre_process_landmark(landmark_list)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    detected_hand_sign = hand_sign_labels[hand_sign_id]
                    
                    current_time = time.time()
                    # Execute the command if the hand sign matches
                    current_time = time.time()
                    if detected_hand_sign in command_mapping and (current_time - last_command_time) > command_delay:
                        st.write(f"Detected hand sign: {detected_hand_sign}")
                        execute_command(command_mapping[detected_hand_sign], delay=command_delay)
                        last_command_time = current_time

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

        cap.release()
        st.write("Webcam stopped.")

if __name__ == "__main__":
    main()