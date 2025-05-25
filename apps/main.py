import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from overlay import overlay_transparent
from utils import eye_aspect_ratio, detect_hand_gesture, draw_healthbar

# Inisialisasi Face Mesh dan Hand Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Load sprite player, peluru, dan shield
player_img = cv2.imread('assets/player_dummy.png', cv2.IMREAD_UNCHANGED)
ammo_img = cv2.imread('assets/ammo_dummy.png', cv2.IMREAD_UNCHANGED)
shield_img = cv2.imread('assets/player_dummy.png', cv2.IMREAD_UNCHANGED)

if player_img is None or ammo_img is None or shield_img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")
if player_img.shape[2] < 4 or ammo_img.shape[2] < 4 or shield_img.shape[2] < 4:
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

# Healthbar awal
health_player1 = 100
health_player2 = 100

# Status shield
shield_active_player1 = False
shield_active_player2 = False

def activate_shield(player_id):
    """
    Activate shield for the specified player.

    Args:
        player_id (str): "Player 1" or "Player 2".
    """
    global shield_active_player1, shield_active_player2
    if player_id == "Player 1":
        shield_active_player1 = True
        print("Player 1 activated SHIELD!")
    elif player_id == "Player 2":
        shield_active_player2 = True
        print("Player 2 activated SHIELD!")

    # Matikan shield setelah beberapa detik
    def deactivate_shield():
        nonlocal player_id
        time.sleep(5)  # Shield aktif selama 5 detik
        if player_id == "Player 1":
            shield_active_player1 = False
        elif player_id == "Player 2":
            shield_active_player2 = False
        print(f"{player_id} shield deactivated.")

    threading.Thread(target=deactivate_shield).start()

def main():
    """
    Main function to run the interactive blink-based shooting game with two players.
    """
    # Buka kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka kamera.")

    # Inisialisasi variabel
    last_blink_time = [0, 0]
    eye_ready_to_blink = [0, 0]
    projectiles = []
    health_player1 = 100
    health_player2 = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        ih, iw = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)
        results_hands = hands.process(rgb_frame)
        current_time = time.time()

        # Reset proyektil yang keluar dari layar
        new_projectiles = []
        for proj in projectiles:
            proj['x'] += 10 if proj['player'] == "Player 1" else -10
            if 0 <= proj['x'] <= iw:
                new_projectiles.append(proj)
        projectiles = new_projectiles

        if results_face.multi_face_landmarks:
            for idx, face_landmarks in enumerate(results_face.multi_face_landmarks[:2]):
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
                    eye_ready_to_blink[idx] = 1

                if blinking and eye_ready_to_blink[idx] == 1 and current_time - last_blink_time[idx] >= BLINK_COOLDOWN:
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
                    last_blink_time[idx] = current_time
                    eye_ready_to_blink[idx] = 0

                # Gambar player
                frame = overlay_transparent(frame, player_img, player_x, player_y, (PLAYER_W, PLAYER_H))

                # Gambar healthbar
                health = health_player1 if idx == 0 else health_player2
                frame = draw_healthbar(frame, f"Player {idx + 1}", health, player_x, player_y - 20)

        # Deteksi gestur tangan
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                gesture = detect_hand_gesture(hand_landmarks)
                if gesture == "thumbs_up":
                    # Aktifkan shield untuk pemain yang melakukan gestur
                    if hand_landmarks.landmark[0].x < 0.5:  # Pemain di kiri frame
                        activate_shield("Player 1")
                    else:  # Pemain di kanan frame
                        activate_shield("Player 2")

        # Gambar proyektil dan deteksi tabrakan
        for proj in projectiles:
            frame = overlay_transparent(frame, ammo_img, proj['x'], proj['y'], (proj['w'], proj['h']))

            # Deteksi tabrakan dengan player
            for idx, face_landmarks in enumerate(results_face.multi_face_landmarks[:2]):
                nose = face_landmarks.landmark[NOSE_IDX]
                player_x = int(nose.x * iw) - PLAYER_W // 2
                player_y = int(nose.y * ih) - PLAYER_H // 2

                if (player_x < proj['x'] < player_x + PLAYER_W and
                    player_y < proj['y'] < player_y + PLAYER_H and
                    proj['player'] != f"Player {idx + 1}" and
                    not (shield_active_player1 if idx == 0 else shield_active_player2)):
                    if idx == 0:
                        health_player1 -= 10
                    else:
                        health_player2 -= 10
                    projectiles.remove(proj)

        # Gambar shield jika aktif
        for idx, face_landmarks in enumerate(results_face.multi_face_landmarks[:2]):
            nose = face_landmarks.landmark[NOSE_IDX]
            player_x = int(nose.x * iw) - PLAYER_W // 2
            player_y = int(nose.y * ih) - PLAYER_H // 2

            if idx == 0 and shield_active_player1:
                frame = overlay_transparent(frame, shield_img, player_x, player_y, (PLAYER_W, PLAYER_H))
            elif idx == 1 and shield_active_player2:
                frame = overlay_transparent(frame, shield_img, player_x, player_y, (PLAYER_W, PLAYER_H))

        # Tampilkan frame
        cv2.imshow("Two Player Blink Shot", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()