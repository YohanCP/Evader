import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from overlay import overlay_transparent
from utils import (
    eye_aspect_ratio,
    detect_hand_gesture,
    draw_healthbar,
)
# Import the new menu manager
import menu_manager

# Constants
RESIZED_FRAME_DIMENSIONS = (640, 480)
MAX_PROJECTILES = 50
SCREEN_CENTER_X = RESIZED_FRAME_DIMENSIONS[0] // 2

# Game States
GAME_STATE_PLAYING = 0
GAME_STATE_WINNER = 1

# Global variables for button rectangles (will be updated in main loop)
replay_button_rect = None
close_button_rect = None

# Initialize Face Mesh and Hand Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)

# Load assets
player1_img_raw = cv2.imread('assets/SPACESHIP/PLAYER 1.png', cv2.IMREAD_UNCHANGED)
player2_img_raw = cv2.imread('assets/SPACESHIP/PLAYER 2.png', cv2.IMREAD_UNCHANGED)
ammo_img_raw = cv2.imread('assets/MAIN UI/SHOT.png', cv2.IMREAD_UNCHANGED)
shield_img = cv2.imread('assets/SHIELD/Shield Smooth Static.png', cv2.IMREAD_UNCHANGED)
winner_img_raw = cv2.imread('assets/MAIN UI/WINNER.png', cv2.IMREAD_UNCHANGED)  # Load winner image
replay_btn_img = cv2.imread('assets/BUTTON TITLE/REPLAY.png', cv2.IMREAD_UNCHANGED)
close_btn_img = cv2.imread('assets/BUTTON TITLE/CLOSE.png', cv2.IMREAD_UNCHANGED)

if player1_img_raw is None or player2_img_raw is None or ammo_img_raw is None or shield_img is None or winner_img_raw is None or replay_btn_img is None or close_btn_img is None:
    raise FileNotFoundError("Gambar tidak ditemukan. Pastikan semua aset ada di direktori yang benar.")
if player1_img_raw.shape[2] < 4 or player2_img_raw.shape[2] < 4 or ammo_img_raw.shape[2] < 4 or shield_img.shape[2] < 4 or winner_img_raw.shape[2] < 4 or replay_btn_img.shape[2] < 4 or close_btn_img.shape[2] < 4:
    raise ValueError("Gambar tidak memiliki alpha channel.")

# Function to rotate images while preserving the alpha channel
def rotate_image_alpha(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

# Rotate spaceship images
player1_rotated = rotate_image_alpha(player1_img_raw, -90)
player2_rotated = rotate_image_alpha(player2_img_raw, 90)

# Eye landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Nose landmark index for player position
NOSE_IDX = 1

# Thresholds and cooldowns
BLINK_THRESHOLD = 0.15
OPEN_THRESHOLD = 0.25
BLINK_COOLDOWN = 1.0

# Player size (using rotated image dimensions)
PLAYER_TARGET_W = 100
PLAYER1_TARGET_H = int(PLAYER_TARGET_W * player1_rotated.shape[0] / player1_rotated.shape[1])
PLAYER2_TARGET_H = int(PLAYER_TARGET_W * player2_rotated.shape[0] / player2_rotated.shape[1])

# Bullet size
AMMO_W = 80
AMMO_H = int(AMMO_W * ammo_img_raw.shape[0] / ammo_img_raw.shape[1])

# Initial health
health_player1 = 100
health_player2 = 100

# Shield status
shield_active_player1 = False
shield_active_player2 = False

# Last shield deactivation times for cooldown
last_shield_deactivation_time_p1 = 0
last_shield_deactivation_time_p2 = 0

# Game control flags for mouse events
_restart_game_flag = False
_exit_game_flag = False

def handle_mouse_event(event, x, y, flags, param):
    global _restart_game_flag, _exit_game_flag, replay_button_rect, close_button_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if replay button was clicked
        if replay_button_rect and \
           x >= replay_button_rect[0] and x <= replay_button_rect[0] + replay_button_rect[2] and \
           y >= replay_button_rect[1] and y <= replay_button_rect[1] + replay_button_rect[3]:
            _restart_game_flag = True
        # Check if close button was clicked
        if close_button_rect and \
           x >= close_button_rect[0] and x <= close_button_rect[0] + close_button_rect[2] and \
           y >= close_button_rect[1] and y <= close_button_rect[1] + close_button_rect[3]:
            _exit_game_flag = True

def activate_shield(player_id):
    global shield_active_player1, shield_active_player2, \
           last_shield_deactivation_time_p1, last_shield_deactivation_time_p2
    current_time = time.time()
    if player_id == "Player 1":
        if not shield_active_player1 and \
           (current_time - last_shield_deactivation_time_p1 >= 5.0):
            shield_active_player1 = True
            threading.Thread(target=deactivate_shield, args=("Player 1",)).start()
    elif player_id == "Player 2":
        if not shield_active_player2 and \
           (current_time - last_shield_deactivation_time_p2 >= 5.0):
            shield_active_player2 = True
            threading.Thread(target=deactivate_shield, args=("Player 2",)).start()

def deactivate_shield(player_id_str):
    global shield_active_player1, shield_active_player2, \
           last_shield_deactivation_time_p1, last_shield_deactivation_time_p2
    time.sleep(3)  # Duration of shield activation
    if player_id_str == "Player 1":
        shield_active_player1 = False
        last_shield_deactivation_time_p1 = time.time()
    elif player_id_str == "Player 2":
        shield_active_player2 = False
        last_shield_deactivation_time_p2 = time.time()

def main():
    global _restart_game_flag, _exit_game_flag, replay_button_rect, close_button_rect
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka kamera.")
    cv2.namedWindow("EVADER")
    cv2.setMouseCallback("EVADER", handle_mouse_event)
    current_game_state = GAME_STATE_PLAYING
    winner_player_id = None
    last_blink_time = [0, 0]
    eye_ready_to_blink = [0, 0]
    projectiles = []
    global health_player1, health_player2, shield_active_player1, shield_active_player2
    health_player1 = 100
    health_player2 = 100
    shield_active_player1 = False
    shield_active_player2 = False
    player_positions = [None, None]
    show_instructions_start_time = time.time()
    instruction_display_duration = 10
    instruction_text = [
        "HOW TO PLAY :",
        "1. Area player dipisah dengan garis tengah.",
        "2. Player 1 (Kiri) aktif dengan jempol kiri.",
        "3. Player 2 (Kanan) aktif dengan jempol kanan.",
        "4. Berkedip untuk menembak.",
        "5. Gerakkan kepala untuk menghindar."
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, RESIZED_FRAME_DIMENSIONS)
        ih, iw = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_face = face_mesh.process(rgb_frame)
        results_hands = hands.process(rgb_frame)
        current_time = time.time()

        if current_game_state == GAME_STATE_PLAYING:
            # Gameplay logic
            new_projectiles = []
            for proj in projectiles:
                proj['x'] += 15 if proj['player'] == "Player 1" else -15
                if 0 <= proj['x'] + proj['w'] and proj['x'] <= iw:
                    new_projectiles.append(proj)
            projectiles = new_projectiles

            if len(projectiles) > MAX_PROJECTILES:
                projectiles.pop(0)

            if results_face.multi_face_landmarks:
                detected_players = []
                for face_landmarks in results_face.multi_face_landmarks[:2]:
                    nose = face_landmarks.landmark[NOSE_IDX]
                    if nose.x < 0 or nose.x > 1 or nose.y < 0 or nose.y > 1:
                        continue

                    # Determine player ID based on nose position relative to the center line
                    player_id = 0 if nose.x * iw < SCREEN_CENTER_X else 1
                    target_player_w = PLAYER_TARGET_W
                    target_player_h = PLAYER1_TARGET_H if player_id == 0 else PLAYER2_TARGET_H

                    # Adjust X position for better placement
                    if player_id == 0:  # Player 1 (left)
                        player_x = int(nose.x * iw) - int(target_player_w * 0.7)
                    else:  # Player 2 (right)
                        player_x = int(nose.x * iw) - int(target_player_w * 0.3)
                    player_y = int(nose.y * ih) - target_player_h // 2

                    # Ensure only one player is detected per side if two faces are detected
                    if player_id not in [p["id"] for p in detected_players]:
                        detected_players.append({
                            "id": player_id,
                            "x": player_x,
                            "y": player_y,
                            "landmarks": face_landmarks
                        })

                # Update player positions
                for player in detected_players:
                    player_id = player["id"]
                    player_positions[player_id] = {"x": player["x"], "y": player["y"]}
                    left_ear = eye_aspect_ratio(player["landmarks"].landmark, LEFT_EYE_IDX, iw, ih)
                    right_ear = eye_aspect_ratio(player["landmarks"].landmark, RIGHT_EYE_IDX, iw, ih)
                    avg_ear = (left_ear + right_ear) / 2.0
                    blinking = avg_ear < BLINK_THRESHOLD
                    eyes_open = avg_ear > OPEN_THRESHOLD

                    if eyes_open:
                        eye_ready_to_blink[player_id] = 1

                    if blinking and eye_ready_to_blink[player_id] == 1 and current_time - last_blink_time[player_id] >= BLINK_COOLDOWN:
                        ammo_rotation_angle = -90 if player_id == 0 else 90
                        rotated_ammo = rotate_image_alpha(cv2.resize(ammo_img_raw, (AMMO_W, AMMO_H), interpolation=cv2.INTER_AREA), ammo_rotation_angle)
                        rotated_ammo_h, rotated_ammo_w = rotated_ammo.shape[:2]

                        # Position the projectile at the tip of the spaceship
                        if player_id == 0:
                            ammo_start_x = player["x"] + PLAYER_TARGET_W - (rotated_ammo_w // 2)
                        else:  # Player 2 shooting to the left
                            ammo_start_x = player["x"] - (rotated_ammo_w // 2)
                        ammo_start_y = player["y"] + (PLAYER1_TARGET_H if player_id == 0 else PLAYER2_TARGET_H) // 2 - (rotated_ammo_h // 2)

                        projectiles.append({
                            'x': ammo_start_x,
                            'y': ammo_start_y,
                            'w': rotated_ammo_w,
                            'h': rotated_ammo_h,
                            'img': rotated_ammo,
                            'player': f"Player {player_id + 1}"
                        })
                        last_blink_time[player_id] = current_time
                        eye_ready_to_blink[player_id] = 0

            # Detect hand gestures
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    gesture = detect_hand_gesture(hand_landmarks)
                    hand_x_normalized = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    if gesture == "thumbs_up":
                        activate_shield("Player 1" if hand_x_normalized < 0.5 else "Player 2")

            # Draw projectiles and detect collisions
            for i in range(len(projectiles) - 1, -1, -1):
                proj = projectiles[i]
                frame = overlay_transparent(frame, proj['img'], proj['x'], proj['y'], (proj['w'], proj['h']))
                collision_occurred = False

                for idx, position in enumerate(player_positions):
                    if position is not None:
                        player_x, player_y = position["x"], position["y"]
                        player_w_check = PLAYER_TARGET_W
                        player_h_check = PLAYER1_TARGET_H if idx == 0 else PLAYER2_TARGET_H

                        # Collision logic
                        player_left = player_x
                        player_top = player_y
                        player_right = player_x + player_w_check
                        player_bottom = player_y + player_h_check

                        proj_left = proj['x']
                        proj_top = proj['y']
                        proj_right = proj['x'] + proj['w']
                        proj_bottom = proj['y'] + proj['h']

                        if (proj_left < player_right and proj_right > player_left and
                            proj_top < player_bottom and proj_bottom > player_top):
                            if proj['player'] != f"Player {idx + 1}":
                                if (idx == 0 and shield_active_player1) or (idx == 1 and shield_active_player2):
                                    collision_occurred = True
                                    break
                                else:
                                    if idx == 0:
                                        health_player1 -= 10
                                    else:
                                        health_player2 -= 10
                                    collision_occurred = True
                                    if health_player1 <= 0:
                                        current_game_state = GAME_STATE_WINNER
                                        winner_player_id = "Player 2"
                                    elif health_player2 <= 0:
                                        current_game_state = GAME_STATE_WINNER
                                        winner_player_id = "Player 1"
                                    break

                if collision_occurred:
                    projectiles.pop(i)

            # Draw shields if active
            for idx, position in enumerate(player_positions):
                if position is not None:
                    player_x, player_y = position["x"], position["y"]
                    player_w_shield = PLAYER_TARGET_W
                    player_h_shield = PLAYER1_TARGET_H if idx == 0 else PLAYER2_TARGET_H

                    shield_scale_factor = 1.3
                    scaled_shield_w = int(player_w_shield * shield_scale_factor)
                    scaled_shield_h = int(player_h_shield * shield_scale_factor)

                    shield_draw_x = player_x - (scaled_shield_w - player_w_shield) // 2
                    shield_draw_y = player_y - (scaled_shield_h - player_h_shield) // 2

                    if idx == 0 and shield_active_player1:
                        frame = overlay_transparent(frame, shield_img, shield_draw_x, shield_draw_y, (scaled_shield_w, scaled_shield_h))
                    elif idx == 1 and shield_active_player2:
                        frame = overlay_transparent(frame, shield_img, shield_draw_x, shield_draw_y, (scaled_shield_w, scaled_shield_h))

        # Draw players and health bars
        for idx, position in enumerate(player_positions):
            if position is not None:
                player_x, player_y = position["x"], position["y"]
                player_to_draw_img = player1_rotated if idx == 0 else player2_rotated
                player_w_draw = PLAYER_TARGET_W
                player_h_draw = PLAYER1_TARGET_H if idx == 0 else PLAYER2_TARGET_H
                frame = overlay_transparent(frame, player_to_draw_img, player_x, player_y, (player_w_draw, player_h_draw))
                health = health_player1 if idx == 0 else health_player2
                frame = draw_healthbar(frame, f"Player {idx + 1}", health, player_x, player_y - 20)

        # Draw dividing line between players
        cv2.line(frame, (SCREEN_CENTER_X, 0), (SCREEN_CENTER_X, ih), (255, 255, 255), 2)

        # Draw instructions with fade-out effect
        elapsed_time = current_time - show_instructions_start_time
        if current_game_state == GAME_STATE_PLAYING and elapsed_time < instruction_display_duration:
            alpha = 1.0
            if elapsed_time > instruction_display_duration - 2:
                alpha = 1.0 - (elapsed_time - (instruction_display_duration - 2)) / 2.0
                alpha = max(0, min(1, alpha))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_base = ih / 480
            font_scale = font_scale_base * 0.5
            thickness = 1
            line_height = int(font_scale * 40)
            max_text_w = 0
            total_text_h = 0

            for line in instruction_text:
                (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                max_text_w = max(max_text_w, text_w)
                total_text_h += (text_h + baseline + 10)

            start_x = (iw - max_text_w) // 2
            start_y = (ih - total_text_h) // 2
            y_offset = start_y

            for line in instruction_text:
                (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                x_pos = start_x + (max_text_w - text_w) // 2
                text_color = (255, 255, 255)
                text_color_alpha = (int(text_color[0] * alpha), int(text_color[1] * alpha), int(text_color[2] * alpha))
                cv2.putText(frame, line, (x_pos, y_offset), font, font_scale, text_color_alpha, thickness, cv2.LINE_AA)
                y_offset += (text_h + baseline + 10)

        # Winner state display
        if current_game_state == GAME_STATE_WINNER:
            projectiles.clear()
            winner_scale_factor = 0.5
            winner_display_w = int(winner_img_raw.shape[1] * winner_scale_factor)
            winner_display_h = int(winner_img_raw.shape[0] * winner_scale_factor)
            winner_scaled_img = cv2.resize(winner_img_raw, (winner_display_w, winner_display_h), interpolation=cv2.INTER_AREA)

            if winner_player_id == "Player 1":
                winner_x = (SCREEN_CENTER_X // 2) - (winner_display_w // 2)
            else:
                winner_x = SCREEN_CENTER_X + (SCREEN_CENTER_X // 2) - (winner_display_w // 2)

            winner_y = int(ih * 0.1)
            frame = overlay_transparent(frame, winner_scaled_img, winner_x, winner_y, (winner_display_w, winner_display_h))

            winner_text = f"{winner_player_id} WINS!"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_scale = ih / 480 * 1.0
            text_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(winner_text, font, text_scale, text_thickness)
            text_x = winner_x + (winner_display_w // 2) - (text_w // 2)
            text_y = winner_y + winner_display_h + 30
            cv2.putText(frame, winner_text, (text_x, text_y), font, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

            # Draw REPLAY and CLOSE buttons
            button_scale_factor = 0.3
            replay_btn_resized = cv2.resize(replay_btn_img, (int(replay_btn_img.shape[1] * button_scale_factor), int(replay_btn_img.shape[0] * button_scale_factor)), interpolation=cv2.INTER_AREA)
            close_btn_resized = cv2.resize(close_btn_img, (int(close_btn_img.shape[1] * button_scale_factor), int(close_btn_img.shape[0] * button_scale_factor)), interpolation=cv2.INTER_AREA)
            replay_btn_w, replay_btn_h = replay_btn_resized.shape[1], replay_btn_resized.shape[0]
            close_btn_w, close_btn_h = close_btn_resized.shape[1], close_btn_resized.shape[0]

            buttons_center_y = int(ih * 0.5)
            total_buttons_width = replay_btn_w + close_btn_w + 40
            start_x_buttons = (iw // 2) - (total_buttons_width // 2)
            replay_x = start_x_buttons
            replay_y = buttons_center_y - (replay_btn_h // 2)
            close_x = start_x_buttons + replay_btn_w + 40
            close_y = buttons_center_y - (close_btn_h // 2)

            frame = overlay_transparent(frame, replay_btn_resized, replay_x, replay_y)
            frame = overlay_transparent(frame, close_btn_resized, close_x, close_y)

            replay_button_rect = (replay_x, replay_y, replay_btn_w, replay_btn_h)
            close_button_rect = (close_x, close_y, close_btn_w, close_btn_h)

        cv2.imshow("EVADER", frame)
        key = cv2.waitKey(1) & 0xFF

        # Global exit (ESC)
        if key == 27:
            _exit_game_flag = True

        # Handle button clicks via mouse callback flags
        if _restart_game_flag:
            current_game_state = GAME_STATE_PLAYING
            winner_player_id = None
            health_player1 = 100
            health_player2 = 100
            shield_active_player1 = False
            shield_active_player2 = False
            projectiles.clear()
            player_positions = [None, None]
            last_blink_time = [0, 0]
            eye_ready_to_blink = [0, 0]
            show_instructions_start_time = time.time()
            _restart_game_flag = False
        if _exit_game_flag:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if menu_manager.run_menu():
        main()