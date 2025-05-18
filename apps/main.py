import cv2
import mediapipe as mp
import numpy as np
import time
from overlay import overlay_transparent
from utils import eye_aspect_ratio, draw_face_bounding_box, draw_face_landmarks

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Load sprite player dan peluru
player_img = cv2.imread('./assets/player_dummy.png', cv2.IMREAD_UNCHANGED)
ammo_img = cv2.imread('./assets/ammo_dummy.png', cv2.IMREAD_UNCHANGED)

if player_img is None or ammo_img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")
if player_img.shape[2] < 4 or ammo_img.shape[2] < 4:
    raise ValueError("Gambar tidak memiliki alpha channel.")

# Indeks landmark mata
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Landmark hidung untuk posisi player
NOSE_IDX = 1

# Threshold dan cooldown
BLINK_THRESHOLD = 0.15
OPEN_THRESHOLD = 0.25
BLINK_COOLDOWN = 1.0

# Ukuran player
PLAYER_W = 100
PLAYER_H = int(PLAYER_W * player_img.shape[0] / player_img.shape[1])

def main():
    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka kamera.")

    # Inisialisasi variabel
    last_blink_time = 0
    eye_ready_to_blink = 0
    projectiles = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        ih, iw = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        current_time = time.time()

        # Reset proyektil yang keluar dari layar
        new_projectiles = []
        for proj in projectiles:
            proj['x'] += 10
            if 0 <= proj['x'] <= iw:
                new_projectiles.append(proj)
        projectiles = new_projectiles

        if results.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results.multi_face_landmarks[:2]):
                # Gambar bounding box dan landmark points
                frame = draw_face_bounding_box(frame, face_landmarks, iw, ih)
                frame = draw_face_landmarks(frame, face_landmarks, iw, ih)

                # Deteksi kedipan mata
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, iw, ih)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, iw, ih)
                avg_ear = (left_ear + right_ear) / 2.0

                nose = face_landmarks.landmark[NOSE_IDX]
                player_x = int(nose.x * iw) - PLAYER_W // 2
                player_y = int(nose.y * ih) - PLAYER_H // 2

                blinking = avg_ear < BLINK_THRESHOLD
                eyes_open = avg_ear > OPEN_THRESHOLD

                if eyes_open:
                    eye_ready_to_blink = 1

                if blinking and eye_ready_to_blink == 1 and current_time - last_blink_time >= BLINK_COOLDOWN:
                    # Tembakkan peluru
                    ammo_w = 30
                    ammo_h = int(ammo_w * ammo_img.shape[0] / ammo_img.shape[1])
                    ammo_start_x = player_x + PLAYER_W if idx == 0 else player_x - ammo_w
                    ammo_start_y = player_y + PLAYER_H // 2 - ammo_h // 2
                    projectiles.append({
                        'x': ammo_start_x,
                        'y': ammo_start_y,
                        'w': ammo_w,
                        'h': ammo_h,
                        'player': f"Player {idx + 1}"
                    })
                    last_blink_time = current_time
                    eye_ready_to_blink = 0

                # Gambar player
                frame = overlay_transparent(frame, player_img, player_x, player_y, (PLAYER_W, PLAYER_H))

        # Gambar proyektil
        for proj in projectiles:
            frame = overlay_transparent(frame, ammo_img, proj['x'], proj['y'], (proj['w'], proj['h']))

        # Tampilkan frame
        cv2.imshow("EVADER", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()