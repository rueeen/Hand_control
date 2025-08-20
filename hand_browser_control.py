import cv2
import time
import math
from collections import deque
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# --------- Parámetros ----------
PINCH_THRESH = 0.06      # Umbral pinch (click)
EMA_ALPHA = 0.35         # Suavizado exponencial (0..1). Mayor = responde más rápido, menor = más suave
WORK_AREA_MARGIN = 0.12  # Margen para mapear la mano a un sub-rectángulo del frame (más margen = más precisión)
SHOW_CURSOR_DOT = True   # Dibuja un punto en el video para depurar la posición del cursor
# -------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def norm_dist(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

def exponential_smooth(prev, new, alpha):
    if prev is None:
        return new
    return (1 - alpha) * prev + alpha * new

def clamp01(x):
    return max(0.0, min(1.0, x))

def main(camera_index=0):
    # Pantalla
    screen_w, screen_h = pyautogui.size()

    # Video
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    # Hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    mode = "mouse"  # "mouse" | "draw"
    print("Controles: [m]=Mouse  [d]=Dibujo  [c]=Clear  [s]=Save  [q/Esc]=Salir")

    # Estados
    last_click_down = False
    smoothed_x = None
    smoothed_y = None

    # Pizarra (para modo dibujo)
    canvas_h, canvas_w = 720, 1280
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    prev_draw_pt = None
    draw_color = (30, 90, 255)   # BGR
    draw_thickness = 4

    # Historial para HUD
    fps = 0.0
    last_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Área de trabajo (recorta bordes para más control)
        mx = int(WORK_AREA_MARGIN * w)
        my = int(WORK_AREA_MARGIN * h)
        work_x0, work_y0 = mx, my
        work_x1, work_y1 = w - mx, h - my
        cv2.rectangle(frame, (work_x0, work_y0), (work_x1, work_y1), (80, 80, 80), 1)

        # Mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pinch = 1.0
        cursor_px = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm = hand.landmark
            wrist = lm[0]
            thumb_tip = lm[4]
            index_tip = lm[8]

            # Distancia pinch
            pinch = norm_dist(thumb_tip, index_tip)

            # Coordenadas normalizadas del índice
            ix_n, iy_n = index_tip.x, index_tip.y

            # Limitar al área de trabajo y normalizar dentro de ella
            ix_n_clamped = clamp01((ix_n - WORK_AREA_MARGIN) / (1 - 2*WORK_AREA_MARGIN))
            iy_n_clamped = clamp01((iy_n - WORK_AREA_MARGIN) / (1 - 2*WORK_AREA_MARGIN))

            # Suavizado
            smoothed_x = exponential_smooth(smoothed_x, ix_n_clamped, EMA_ALPHA)
            smoothed_y = exponential_smooth(smoothed_y, iy_n_clamped, EMA_ALPHA)

            # Mapeo a pantalla y a lienzo
            cur_x = int(smoothed_x * screen_w)
            cur_y = int(smoothed_y * screen_h)
            cursor_px = (int(ix_n * w), int(iy_n * h))  # para dibujar un punto en el frame

            # HUD Mediapipe
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, f"pinch={pinch:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)

            if mode == "mouse":
                # Mover cursor
                try:
                    pyautogui.moveTo(cur_x, cur_y)
                except Exception:
                    pass

                # Click/Drag con pinch
                is_down = (pinch < PINCH_THRESH)
                if is_down and not last_click_down:
                    pyautogui.mouseDown()
                    last_click_down = True
                elif not is_down and last_click_down:
                    pyautogui.mouseUp()
                    last_click_down = False

            elif mode == "draw":
                # Convertir a coords de lienzo
                draw_x = int(smoothed_x * canvas_w)
                draw_y = int(smoothed_y * canvas_h)
                is_drawing = (pinch < PINCH_THRESH)

                if is_drawing:
                    if prev_draw_pt is None:
                        prev_draw_pt = (draw_x, draw_y)
                    else:
                        cv2.line(canvas, prev_draw_pt, (draw_x, draw_y), draw_color, draw_thickness, cv2.LINE_AA)
                        prev_draw_pt = (draw_x, draw_y)
                else:
                    prev_draw_pt = None

        else:
            # Sin mano: soltar mouse si quedó pulsado
            if last_click_down:
                pyautogui.mouseUp()
                last_click_down = False
            prev_draw_pt = None

        # FPS
        now = time.time()
        dt = now - last_t
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_t = now

        # Overlays
        cv2.putText(frame, f"Mode: {mode.upper()}  FPS:{fps:.1f}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if SHOW_CURSOR_DOT and cursor_px is not None:
            cv2.circle(frame, cursor_px, 8, (0, 255, 255), -1)

        # Ventanas
        cv2.imshow("Camera (q/Esc para salir, m=mouse, d=dibujo, c=clear, s=save)", frame)
        if mode == "draw":
            cv2.imshow("Pizarra (dibujo)", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('m'):
            mode = "mouse"
            prev_draw_pt = None
        elif key == ord('d'):
            mode = "draw"
        elif key == ord('c') and mode == "draw":
            canvas[:] = 255
        elif key == ord('s') and mode == "draw":
            fname = f"dibujo_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print("Guardado:", fname)

    # Cleanup
    try:
        if last_click_down:
            pyautogui.mouseUp()
    except Exception:
        pass

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(0)
