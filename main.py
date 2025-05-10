import cv2
import mediapipe as mp
import numpy as np
import time

# Inisialisasi Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Load sprite player dan peluru
player_img = cv2.imread('player 1.png', cv2.IMREAD_UNCHANGED)
ammo_img = cv2.imread('ammo1.png', cv2.IMREAD_UNCHANGED)

if player_img is None or ammo_img is None:
    raise FileNotFoundError("Gambar tidak ditemukan.")
if player_img.shape[2] < 4 or ammo_img.shape[2] < 4:
    raise ValueError("Gambar tidak memiliki alpha channel.")

def overlay_transparent(bg, overlay, x, y, overlay_size=None):
    bg = bg.copy()
    if overlay_size:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x + w > bg.shape[1] or y + h > bg.shape[0] or x < 0 or y < 0:
        return bg
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    mask = np.stack([mask]*3, axis=-1)
    roi = bg[y:y+h, x:x+w]
    blended = (1.0 - mask) * roi + mask * overlay_img
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    return bg

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    p = [landmarks[i] for i in eye_indices]
    p = [(int(pt.x * img_w), int(pt.y * img_h)) for pt in p]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    ear = (A + B) / (2.0 * C)
    return ear

# Indeks landmark mata
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Landmark hidung untuk posisi player
NOSE_IDX = 1

# Threshold dan cooldown
blink_threshold = 0.15
open_threshold = 0.25
last_blink_time = 0
blink_cooldown = 1.0
eye_ready_to_blink = 0  # 0: belum siap, 1: sudah melek penuh dan siap kedip

# List proyektil aktif
projectiles = []

# Ukuran player
player_w = 100
player_h = int(player_w * player_img.shape[0] / player_img.shape[1])

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    ih, iw = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    current_time = time.time()

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_IDX, iw, ih)
        right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_IDX, iw, ih)
        avg_ear = (left_ear + right_ear) / 2.0

        # Posisi player berdasarkan hidung
        nose = face_landmarks.landmark[NOSE_IDX]
        player_x = int(nose.x * iw) - player_w // 2
        player_y = int(nose.y * ih) - player_h // 2

        blinking = avg_ear < blink_threshold
        eyes_open = avg_ear > open_threshold

        if eyes_open:
            eye_ready_to_blink = 1  # Reset status: siap deteksi kedipan

        if blinking and eye_ready_to_blink == 1 and current_time - last_blink_time >= blink_cooldown:
            # Tembakkan peluru
            ammo_w = 30
            ammo_h = int(ammo_w * ammo_img.shape[0] / ammo_img.shape[1])
            ammo_start_x = player_x + player_w
            ammo_start_y = player_y + player_h // 2 - ammo_h // 2
            projectiles.append({'x': ammo_start_x, 'y': ammo_start_y, 'w': ammo_w, 'h': ammo_h})
            last_blink_time = current_time
            eye_ready_to_blink = 0  # Tunggu sampai mata melek penuh lagi

    # Gambar player
    frame = overlay_transparent(frame, player_img, player_x, player_y, (player_w, player_h))

    # Perbarui dan gambar proyektil
    new_projectiles = []
    for proj in projectiles:
        proj['x'] += 10
        if proj['x'] < iw:
            frame = overlay_transparent(frame, ammo_img, proj['x'], proj['y'], (proj['w'], proj['h']))
            new_projectiles.append(proj)
    projectiles = new_projectiles

    # Tampilkan frame
    cv2.imshow("Player Blink Shot (No Sound)", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()