import numpy as np
import cv2

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    p = [landmarks[i] for i in eye_indices]
    p = [(int(pt.x * img_w), int(pt.y * img_h)) for pt in p]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear

def draw_face_bounding_box(frame, face_landmarks, img_width, img_height):
    x_coords = [int(landmark.x * img_width) for landmark in face_landmarks.landmark]
    y_coords = [int(landmark.y * img_height) for landmark in face_landmarks.landmark]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return frame

def draw_face_landmarks(frame, face_landmarks, img_width, img_height):
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame