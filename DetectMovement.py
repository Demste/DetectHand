import mediapipe as mp
import cv2
import numpy as np
import time

# Load Mediapipe Pose module
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.3)
pose_drawer = mp.solutions.drawing_utils
# Initialize OpenCV's HOG-based people detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


path="6077710-uhd_3840_2160_25fps.mp4"
camera = cv2.VideoCapture(path)

log_file_path = 'pose_log.txt'

def log_event(message):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")} - {message}\n')

def Check_Hand_Raised(frame, landmarks, mp_pose):
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
    
    # Sağ elin kaldırılıp kaldırılmadığını kontrol etme
    if right_shoulder > right_wrist:
        cv2.putText(frame, 'Right Hand Raised!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        log_event('Right Hand Raised')

    
    # Sol elin kaldırılıp kaldırılmadığını kontrol etme
    if left_shoulder > left_wrist:
        cv2.putText(frame, 'Left Hand Raised!', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        log_event('Left Hand Raised')


last_detection_time = 0
hand_speed_threshold = 0.2  # Eşik değer
min_distance_threshold = 0.1  # Minimum mesafe eşiği


def Check_Applause(frame, landmarks, mp_pose):
    global  hand_speed_threshold, min_distance_threshold, hands_together
    
    # Sağ ve sol el landmark noktalarını al
    right_hand_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hand_landmark = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    
    # Sağ ve sol el landmark noktalarının pozisyonlarını al
    right_hand_x = right_hand_landmark.x
    right_hand_y = right_hand_landmark.y
    left_hand_x = left_hand_landmark.x
    left_hand_y = left_hand_landmark.y
    
    # Sağ ve sol el landmark noktaları arasındaki mesafeyi hesapla
    distance_between_hands = abs(right_hand_x - left_hand_x) + abs(right_hand_y - left_hand_y)
    
    # Eller birbirine çok yakınsa
    if distance_between_hands < min_distance_threshold:
        cv2.putText(frame, 'Eller değdi! ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        log_event('Eller değdi')



while camera.isOpened():
    # Read a frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Failed to read frame from camera.")
        continue

    # Convert the frame to RGB format for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    result = pose_detector.process(frame_rgb)

    # Check if the left hand is raised
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark 
        Check_Hand_Raised(frame, landmarks, mp_pose)
        Check_Applause(frame,landmarks,mp_pose)

    # Draw pose landmarks on the frame
    pose_drawer.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
pose_detector.close()



