import cv2
import numpy as np
from overlay import overlay_transparent

def run_menu():
    # Load assets
    title_img = cv2.imread('assets/BUTTON TITLE/TITLE.png', cv2.IMREAD_UNCHANGED)
    start_btn_img = cv2.imread('assets/BUTTON TITLE/START.png', cv2.IMREAD_UNCHANGED)

    if title_img is None or start_btn_img is None:
        raise FileNotFoundError("Gambar tidak ditemukan. Pastikan semua aset ada di direktori yang benar.")
    if title_img.shape[2] < 4 or start_btn_img.shape[2] < 4:
        raise ValueError("Gambar tidak memiliki alpha channel.")

    # Constants
    RESIZED_FRAME_DIMENSIONS = (640, 480)
    SCREEN_CENTER_X = RESIZED_FRAME_DIMENSIONS[0] // 2

    # Initialize variables
    start_button_rect = None
    _start_game_flag = False

    def handle_mouse_event(event, x, y, flags, param):
        nonlocal _start_game_flag
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if start button was clicked
            if start_button_rect and \
               x >= start_button_rect[0] and x <= start_button_rect[0] + start_button_rect[2] and \
               y >= start_button_rect[1] and y <= start_button_rect[1] + start_button_rect[3]:
                _start_game_flag = True

    # Main menu loop
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Tidak dapat membuka kamera.")
    cv2.namedWindow("EVADER")
    cv2.setMouseCallback("EVADER", handle_mouse_event)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, RESIZED_FRAME_DIMENSIONS)
        ih, iw = frame.shape[:2]

        # Draw title screen
        title_scaled = cv2.resize(title_img, (int(iw * 0.8), int(ih * 0.8)), interpolation=cv2.INTER_AREA)
        title_x = (iw - title_scaled.shape[1]) // 2
        title_y = int(ih * 0.2)
        frame = overlay_transparent(frame, title_scaled, title_x, title_y, (title_scaled.shape[1], title_scaled.shape[0]))

        # Draw start button
        start_btn_resized = cv2.resize(start_btn_img, (int(start_btn_img.shape[1] * 0.3), int(start_btn_img.shape[0] * 0.3)), interpolation=cv2.INTER_AREA)
        start_btn_w, start_btn_h = start_btn_resized.shape[1], start_btn_resized.shape[0]
        start_x = (iw // 2) - (start_btn_w // 2)
        start_y = int(ih * 0.7)
        frame = overlay_transparent(frame, start_btn_resized, start_x, start_y, (start_btn_w, start_btn_h))
        start_button_rect = (start_x, start_y, start_btn_w, start_btn_h)

        cv2.imshow("EVADER", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit on ESC
        if key == 27:
            break

        # Check if START button was clicked
        if _start_game_flag:
            break

    cap.release()
    cv2.destroyAllWindows()

    return _start_game_flag