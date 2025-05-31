import numpy as np
import cv2

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    p = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices])
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_hand_gesture(hand_landmarks, thumb_threshold=0.1):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]

    if (thumb_tip.y + thumb_threshold < index_tip.y) and (thumb_tip.y + thumb_threshold < middle_tip.y):
        return "thumbs_up"

    return None

def draw_healthbar(frame, player_id, health, x, y, w=100, h=10, bg_color=(0, 0, 0), fg_color=(0, 255, 0)):
    # Draw background of the healthbar
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, 2)

    # Draw filled part of the healthbar
    fill_width = int((health / 100) * w)
    cv2.rectangle(frame, (x, y), (x + fill_width, y + h), fg_color, -1)

    # Add player label
    cv2.putText(frame, player_id, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame