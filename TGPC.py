import csv
import copy
from collections import Counter
from collections import deque

import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
import streamlit as st
from streamlit.components.v1 import html
from PIL import Image
import app as ap
import subprocess
import time

def main():
    st.title("Hand Sign Command Control")
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
        st.sidebar.write("No existing hand sign commands found.")
        
    cap_device = st.selectbox("Select Camera",["Web Cam", "External Cam"])
    if cap_device == "Web Cam":
        cap_device = 0
    elif cap_device == "External Cam":
        cap_device = 2
        
       
    cap_width = st.slider("Camera Width", min_value=10, max_value=1920, value=1280,step=10)
    cap_height = st.slider("Camera Height", min_value=10, max_value=1080, value=720,step=10)
    isFlip = st.checkbox("Flip Image", value=True)
    use_static_image_mode = st.checkbox("Use Static Image Mode", value=False)
    use_brect = st.checkbox("Use Bounding Rect", value=True)
    min_detection_confidence = st.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    min_tracking_confidence = st.slider("Min Tracking Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    max_num_hands = st.slider("Max Number of Hands", min_value=1, max_value=10, value=2, step=1)
    start_webcam = st.button("Start Webcam")
    if start_webcam and start_webcam == True:
        st.write("Webcam started.")
        video_placeholder = st.empty()
        
        start_webcam = False
        stop_webcam = st.button("Stop Webcam", key="stop_webcam")
        use_brect = True

        # Camera preparation ###############################################################
        cap = cv.VideoCapture(cap_device) # type: ignore
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width) # type: ignore
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height) # type: ignore

        # Model load #############################################################
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        point_history_classifier = PointHistoryClassifier()

        # Read labels ###########################################################
        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                          encoding='utf-8-sig') as f:
                keypoint_classifier_labels = csv.reader(f)
                keypoint_classifier_labels = [
                        row[0] for row in keypoint_classifier_labels
                ]
        with open(
                        'model/point_history_classifier/point_history_classifier_label.csv',
                        encoding='utf-8-sig') as f:
                point_history_classifier_labels = csv.reader(f)
                point_history_classifier_labels = [
                        row[0] for row in point_history_classifier_labels
                ]


        # FPS Measurement ########################################################
        cvFpsCalc = CvFpsCalc(buffer_len=10)

        # Coordinate history #################################################################
        history_length = 16
        point_history = deque(maxlen=history_length)

        # Finger gesture history ################################################
        finger_gesture_history = deque(maxlen=history_length)

        #  ########################################################################
        
        mode = 0 # Default mode
        
        command_cooldown = 5
        # Dictionary to track the last execution time for each hand sign
        last_execution_time = {}
        while True:
                fps = cvFpsCalc.get()

                # Process Key (ESC: end) #################################################
                key = cv.waitKey(10) # type: ignore
                if key == 27:  # ESC
                        break
                number, mode = ap.select_mode(key, mode)

                # Camera capture #####################################################
                ret, image = cap.read()
                if not ret:
                        break
                if isFlip==True:
                        image = cv.flip(image, 1) # type: ignore
                else:
                        image = cv.flip(image, 0)

                # Detection implementation #############################################################
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB) # type: ignore
                debug_image = copy.deepcopy(image)

                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True

                # Landmark detection
                if results.multi_hand_landmarks is not None:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                                                                  results.multi_handedness):
                                # Bounding box calculation
                                print("Hand landmark:"+str(hand_landmarks))
                                brect = ap.calc_bounding_rect(debug_image, hand_landmarks)
                                print("Bounding Box"+str(brect))
                                
                                # Landmark calculation
                                print("Hand landmarks"+str(hand_landmarks))
                                landmark_list = ap.calc_landmark_list(debug_image, hand_landmarks)
                                print("Landmark List"+str(landmark_list))
                                
                                
                                # Conversion to relative coordinates / normalized coordinates
                                print("Landmark List"+str(landmark_list))
                                pre_processed_landmark_list = ap.pre_process_landmark(landmark_list)
                                print("Pre-processed Landmark List"+str(pre_processed_landmark_list))
                                
                                print("point_history"+str(point_history))
                                pre_processed_point_history_list = ap.pre_process_point_history(debug_image, point_history)
                                print("Pre-processed Point History List"+str(pre_processed_point_history_list))
                                
                                
                                # Write to the dataset file
                                ap.logging_csv(number, mode, pre_processed_landmark_list,
                                                        pre_processed_point_history_list)

                                # Hand sign classification
                            
                                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                                
                          
                                detected_hand_sign = hand_sign_labels[hand_sign_id]
                   
                                
                               
                                if hand_sign_id == 2:  # Point gesture
                                        point_history.append(landmark_list[8])
                                else:
                                        point_history.append([0, 0])

                                # Finger gesture classification
                                finger_gesture_id = 0
                                point_history_len = len(pre_processed_point_history_list)
                                if point_history_len == (history_length * 2):
                                        finger_gesture_id = point_history_classifier(
                                                pre_processed_point_history_list)

                                # Calculates the gesture IDs in the latest detection
                                finger_gesture_history.append(finger_gesture_id)
                                most_common_fg_id = Counter(finger_gesture_history).most_common()
                                
                                #############################
                                
                                # Drawing part
                                debug_image = ap.draw_bounding_rect(use_brect, debug_image, brect)
                                
                                debug_image = ap.draw_landmarks(debug_image, landmark_list)
                                
                                debug_image = ap.draw_info_text(
                                        debug_image,
                                        brect,
                                        handedness,
                                        keypoint_classifier_labels[hand_sign_id],
                                        point_history_classifier_labels[most_common_fg_id[0][0]],
                                )
                                current_time = time.time()
                                last_time = last_execution_time.get(detected_hand_sign, 0)
                                if detected_hand_sign in command_mapping and (current_time - last_time > command_cooldown):
                                        st.write(f"Detected hand sign: {detected_hand_sign}")
                                        execute_command(command_mapping[detected_hand_sign])
                                        last_execution_time[detected_hand_sign] = current_time
                                elif detected_hand_sign in command_mapping:
                                        print(f"Detected hand sign: {detected_hand_sign} (Cooldown in effect)")
                                else:
                                        print(f"Detected hand sign: {detected_hand_sign} (No command assigned)")
                                # print(keypoint_classifier_labels[hand_sign_id])
                                # print(point_history_classifier_labels[most_common_fg_id[0][0]])
                else:
                        point_history.append([0, 0])

                debug_image = ap.draw_point_history(debug_image, point_history)
                debug_image = ap.draw_info(debug_image, fps, mode, number)

                # Display the image in Streamlit##############################################
                frame=Image.fromarray(debug_image)
                video_placeholder.image(frame, caption="Webcam Feed", use_container_width=True)
                # Stop the webcam feed stop button is pressed
                if stop_webcam:
                    st.write("Stopping webcam...")
                    stop_webcam = False
                    video_placeholder.empty()
                    break


        st.write("Webcam stopped.")
        # Release the camera and close all OpenCV windows
        cap.release()
        cv.destroyAllWindows() # type: ignore
        
def execute_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Command executed successfully: {command}")
        print(f"Output: {result.stdout}")
          # Add a delay after executing the command
    except subprocess.CalledProcessError as e:
        st.error(f"Error while executing command: {e.stderr}")
          # Add a delay even if the command fails


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
        
if __name__ == "__main__":     
        main()