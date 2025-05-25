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

def detect_hand_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]

    # Contoh: Deteksi thumbs up
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "thumbs_up"

    return None

def draw_healthbar(frame, player_id, health, x, y, w=100, h=10):
    # Draw background of the healthbar
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # Draw filled part of the healthbar
    fill_width = int((health / 100) * w)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + h), (0, 255, 0), -1)
    # Add player label
    cv2.putText(frame, player_id, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame