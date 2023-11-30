import os

shape_predictor_path = os.path.join('C:/Users/91908/Downloads/', 'shape_predictor_68_face_landmarks.dat')
output_file_path = r'driver_data.csv'

EYE_DROWSINESS_THRESHOLD = 0.20
EYE_DROWSINESS_INTERVAL = 1.7
MOUTH_DROWSINESS_THRESHOLD = 0.37
MOUTH_DROWSINESS_INTERVAL = 1.0
DISTRACTION_INTERVAL = 2.5
NORMAL_INTERVAL = 1.0
EMERGENCY_BRAKE_THRESHOLD = 25
