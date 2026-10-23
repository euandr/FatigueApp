

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==============================
# CONFIG
# EMA base (mais alto = mais suave/slower). Use ALPHA_FAST for grandes variações.
ALPHA = 0.90
ALPHA_FAST = 0.6
SMOOTH_ADAPT_THRESHOLD = 0.08
THRESH = 0.06

smooth_score = 0
smooth_initialized = False

# ==============================
# FUNÇÃO DE POSTURA
def posture_score(lm):

    nose = lm[mp_holistic.PoseLandmark.NOSE]
    left_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    # centro dos ombros
    shoulder_y = (left_sh.y + right_sh.y) / 2
    shoulder_z = (left_sh.z + right_sh.z) / 2

    # escala do corpo (normalização)
    shoulder_width = abs(left_sh.x - right_sh.x)

    # tentar usar altura do torso como alternativa (mais robusto para corpos volumosos)
    left_hip = lm[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = lm[mp_holistic.PoseLandmark.RIGHT_HIP]
    torso_height = abs(((left_sh.y + right_sh.y) / 2) - ((left_hip.y + right_hip.y) / 2))

    # usa a maior dimensão disponível para normalizar (evita valores muito pequenos)
    scale = max(shoulder_width, torso_height)

    if scale < 1e-6:
        return None

    # ==========================
    # COMPONENTE 1: Y (principal)
    # ==========================
    pitch_y = (nose.y - shoulder_y) / scale

    # ==========================
    # COMPONENTE 2: Z (auxiliar)
    # ==========================
    depth_z = (shoulder_z - nose.z) / scale

    # ==========================
    # SCORE FINAL (junção)
    # ==========================
    score = pitch_y + 0.5 * depth_z

    return score

# 2. Inclinação da cabeça (pescoço torto)
def Inclinacao_cabeca(landmarks):
    left_ear = landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]
    
    DiferencaEarAltura = abs(left_ear.y - right_ear.y)
    inclinada= False
    if DiferencaEarAltura > 0.07:  
        inclinada = True
    return {"diferenca": DiferencaEarAltura,  "inclinacao": inclinada}










import threading
import queue

# -----------------------------
# Threaded pipeline: capture -> process -> display
# -----------------------------
capture_q = queue.Queue(maxsize=4)
display_q = queue.Queue(maxsize=4)
stop_event = threading.Event()

cap = cv2.VideoCapture(0)

def capture_loop():
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            capture_q.put_nowait(frame)
        except queue.Full:
            # fila cheia: descarta frame mais antigo implicitamente
            pass
    try:
        cap.release()
    except Exception:
        pass

def processing_loop():
    global smooth_score, smooth_initialized
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while not stop_event.is_set():
            try:
                frame = capture_q.get(timeout=0.1)
            except queue.Empty:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                score = posture_score(lm)

                if score is not None:
                    if not smooth_initialized:
                        smooth_score = score
                        smooth_initialized = True
                    else:
                        diff = abs(score - smooth_score)
                        alpha = ALPHA_FAST if diff > SMOOTH_ADAPT_THRESHOLD else ALPHA
                        smooth_score = alpha * smooth_score + (1 - alpha) * score

                    inclinacao_info = Inclinacao_cabeca(lm)

                    if inclinacao_info["inclinacao"]:
                        status = "postura ruim"
                        color = (0, 0, 255)
                    elif smooth_score > THRESH:
                        status = "postura ruim"
                        color = (0, 0, 255)
                    else:
                        status = "POSTURA OK"
                        color = (0, 255, 0)

                    cv2.putText(frame, f"ScoreFrente: {smooth_score:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, f"ScoreBruto: {score:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    cv2.putText(frame, f"ScoreLados: {inclinacao_info['diferenca']:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame, status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # desenha landmarks (se houver)
            try:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            except Exception:
                pass

            try:
                display_q.put_nowait(frame)
            except queue.Full:
                pass

def start_threads():
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=processing_loop, daemon=True)
    t_cap.start()
    t_proc.start()
    return t_cap, t_proc

threads = start_threads()

try:
    while True:
        try:
            disp = display_q.get(timeout=0.1)
        except queue.Empty:
            # nada pronto ainda
            if stop_event.is_set():
                break
            continue

        cv2.imshow("Postura - Y + Z Fusion", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break
finally:
    stop_event.set()
    # aguarda threads (são daemon, mas join seguro)
    for t in threads:
        try:
            t.join(timeout=1.0)
        except Exception:
            pass
    cv2.destroyAllWindows()