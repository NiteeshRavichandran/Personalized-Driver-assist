from parameters import *
from scipy.spatial import distance
from imutils import face_utils as face
import imutils
import time
import dlib
import cv2
from datetime import datetime


def get_max_area_rect(rects):
    if len(rects) == 0: return
    areas = []
    for rect in rects:
        areas.append(rect.area())
    return rects[areas.index(max(areas))]


def get_eye_aspect_ratio(eye):
    # eye landmarks (x, y)-coordinates
    vertical_1 = distance.euclidean(eye[1], eye[5])
    vertical_2 = distance.euclidean(eye[2], eye[4])
    horizontal = distance.euclidean(eye[0], eye[3])
    return (vertical_1 + vertical_2) / (horizontal * 2)


# computes the mouth aspect ratio (mar)
def get_mouth_aspect_ratio(mouth):
    horizontal = distance.euclidean(mouth[0], mouth[4])
    vertical = 0
    for coord in range(1, 4):
        vertical += distance.euclidean(mouth[coord], mouth[8 - coord])
    return vertical / (horizontal * 3)


def facial_processing():
    distracton_initialized = False
    eye_initialized = False
    mouth_initialized = False
    normal_initialized = False

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('C:/Users/91908/Downloads/shape_predictor_68_face_landmarks.dat')

    ls, le = face.FACIAL_LANDMARKS_IDXS["left_eye"]
    rs, re = face.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    fps_counter = 0
    fps_to_display = 'initializing...'
    fps_timer = time.time()
    while True:
        _, frame = cap.read()
        fps_counter += 1
        frame = cv2.flip(frame, 1)
        if time.time() - fps_timer >= 1.0:
            fps_to_display = fps_counter
            fps_timer = time.time()
            fps_counter = 0
        cv2.putText(frame, "FPS :" + str(fps_to_display), (frame.shape[1] - 100, frame.shape[0] - 10), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        rect = get_max_area_rect(rects)

        if rect != None:
            if distracton_initialized == True:
                interval = time.time() - distracton_start_time
                interval = str(round(interval, 3))
                dateTime = datetime.now()
                distracton_initialized = False
                info = "Date: " + str(dateTime) + ", Interval: " + interval + ", Type: Eyes not on road"
                info = info + "\n"
                if time.time() - distracton_start_time > DISTRACTION_INTERVAL:
                    with open(r'C:/Users/91908/Downloads/output.txt', "a+") as file_object:
                        file_object.write(info)

            shape = predictor(gray, rect)
            shape = face.shape_to_np(shape)

            leftEye = shape[ls:le]
            rightEye = shape[rs:re]
            leftEAR = get_eye_aspect_ratio(leftEye)
            rightEAR = get_eye_aspect_ratio(rightEye)

            inner_lips = shape[60:68]
            mar = get_mouth_aspect_ratio(inner_lips)

            eye_aspect_ratio = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)
            lipHull = cv2.convexHull(inner_lips)
            cv2.drawContours(frame, [lipHull], -1, (255, 255, 255), 1)

            cv2.putText(frame, "EAR: {:.2f} MAR{:.2f}".format(eye_aspect_ratio, mar), (10, frame.shape[0] - 10), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if eye_aspect_ratio < EYE_DROWSINESS_THRESHOLD:

                if not eye_initialized:
                    eye_start_time = time.time()
                    eye_initialized = True
                if time.time() - eye_start_time >= EYE_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU ARE DROWSY!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                if eye_initialized == True:
                    interval_eye = time.time() - eye_start_time
                    interval_eye = str(round(interval_eye, 3))
                    dateTime_eye = datetime.now()
                    eye_initialized = False
                    info_eye = "Date: " + str(dateTime_eye) + ", Interval: " + interval_eye + ", Type:Drowsy"
                    info_eye = info_eye + "\n"
                    if time.time() - eye_start_time >= EYE_DROWSINESS_INTERVAL:
                        # store info into a txt file
                        with open(r'C:/Users/91908/Downloads/output.txt', "a+") as file_object:
                            file_object.write(info_eye)

            if mar > MOUTH_DROWSINESS_THRESHOLD:

                if not mouth_initialized:
                    mouth_start_time = time.time()
                    mouth_initialized = True
                # checks if the user is yawning for a sufficient number of frames
                if time.time() - mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                    cv2.putText(frame, "YOU ARE YAWNING!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                if mouth_initialized == True:
                    interval_mouth = time.time() - mouth_start_time
                    interval_mouth = str(round(interval_mouth, 3))
                    dateTime_mouth = datetime.now()
                    mouth_initialized = False
                    info_mouth = "Date: " + str(dateTime_mouth) + ", Interval: " + interval_mouth + ", Type:Yawning"
                    info_mouth = info_mouth + "\n"
                    if time.time() - mouth_start_time >= MOUTH_DROWSINESS_INTERVAL:
                        with open(r'C:/Users/91908/Downloads/output.txt', "a+") as file_object:
                            file_object.write(info_mouth)

            if (eye_initialized == False) & (mouth_initialized == False) & (distracton_initialized == False):

                if not normal_initialized:
                    normal_start_time = time.time()
                    normal_initialized = True

                if time.time() - normal_start_time >= NORMAL_INTERVAL:
                    cv2.putText(frame, "Normal!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print('Normal')
            else:
                if normal_initialized == True:
                    interval_normal = time.time() - normal_start_time
                    interval_normal = str(round(interval_normal, 3))
                    dateTime_normal = datetime.now()
                    normal_initialized = False
                    info_normal = "Date: " + str(dateTime_normal) + ", Interval: " + interval_normal + ", Type:Normal"
                    info_normal = info_normal + "\n"
                    if time.time() - normal_start_time >= NORMAL_INTERVAL:
                        with open(r'C:/Users/91908/Downloads/output.txt', "a+") as file_object:
                            file_object.write(info_normal)


        else:
            if eye_initialized == True:
                interval_eye = time.time() - eye_start_time
                interval_eye = str(round(interval_eye, 3))
                dateTime_eye = datetime.now()
                eye_initialized = False
                info_eye = "Date: " + str(dateTime_eye) + ", Interval: " + interval_eye + ", Type:Drowsy"
                info_eye = info_eye + "\n"
                if time.time() - eye_start_time >= EYE_DROWSINESS_INTERVAL:
                    with open(r'C:/Users/91908/Downloads/output.txt', "a+") as file_object:
                        file_object.write(info_eye)

            if not distracton_initialized:
                distracton_start_time = time.time()
                distracton_initialized = True
            if time.time() - distracton_start_time > DISTRACTION_INTERVAL:
                cv2.putText(frame, "YOU ARE DISTRACTED PLEASE KEEP EYES ON ROAD", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5) & 0xFF


        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    facial_processing()