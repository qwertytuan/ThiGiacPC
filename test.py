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

def main():
    st.title("Hand Sign")
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
        mode = 0

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
                                brect = ap.calc_bounding_rect(debug_image, hand_landmarks)
                                # Landmark calculation
                                landmark_list = ap.calc_landmark_list(debug_image, hand_landmarks)

                                # Conversion to relative coordinates / normalized coordinates
                                pre_processed_landmark_list = ap.pre_process_landmark(landmark_list)
                                pre_processed_point_history_list = ap.pre_process_point_history(debug_image, point_history)
                                # Write to the dataset file
                                ap.logging_csv(number, mode, pre_processed_landmark_list,
                                                        pre_processed_point_history_list)

                                # Hand sign classification
                                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                                print(hand_sign_id)
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
        
        
if __name__ == "__main__":     
        main()