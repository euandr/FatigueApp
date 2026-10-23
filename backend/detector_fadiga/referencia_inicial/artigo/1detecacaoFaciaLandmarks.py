import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from threading import Thread
import time

# Constantes
# ALARM = r'C:\Users\andre\Documents\FatigueDetector\detector_fadiga\alarm-clock-90867.mp3'
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
EAR_HISTORY = []
MOUTH_AR_THRESH = 0.6
COUNTER = 0
BLINK_COUNTER = 0
TOTAL_BLINKS = 0
ALARM_ON = False
YAWN_ON = False

# Alarme
# def sound_alarm(path=ALARM):
#     playsound.playsound(path)

# EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[1])
    B = dist.euclidean(mouth[2], mouth[3])
    if B == 0:
        return 0.0
    return A / B

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Índices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]

# câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, int(frame.shape[0] * 800 / frame.shape[1])))
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # conversão dos pontos
        left_eye = np.array([(int(mesh_points[i].x * w), int(mesh_points[i].y * h)) for i in LEFT_EYE_IDX])
        right_eye = np.array([(int(mesh_points[i].x * w), int(mesh_points[i].y * h)) for i in RIGHT_EYE_IDX])
        mouth = np.array([(int(mesh_points[i].x * w), int(mesh_points[i].y * h)) for i in MOUTH_IDX])

        # EAR e MAR
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        EAR_HISTORY.append(ear)
        if len(EAR_HISTORY) > 30:
            EAR_HISTORY.pop(0)

        # =========================
        # 🔵 DESENHO DOS LANDMARKS
        # =========================

        # olhos (pontos verdes)
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # boca (pontos azuis)
        for (x, y) in mouth:
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # opcional: rosto inteiro (cinza leve)
        for lm in mesh_points:
            x = int(lm.x * w)
            y = int(lm.y * h)
            # cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)

        # =========================
        # lógica de piscada e fadiga
        # =========================

        blinks_in_window = 0
        for i in range(1, len(EAR_HISTORY)):
            if EAR_HISTORY[i-1] > EYE_AR_THRESH and EAR_HISTORY[i] < EYE_AR_THRESH:
                blinks_in_window += 1

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            BLINK_COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                ALARM_ON = True
                # Thread(target=sound_alarm, daemon=True).start()
                cv2.putText(frame, "[ALERTA] FADIGA!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if BLINK_COUNTER >= 3:
                TOTAL_BLINKS += 1
                BLINK_COUNTER = 0
        else:
            COUNTER = 0
            ALARM_ON = False
            BLINK_COUNTER = 0

        if blinks_in_window >= 5:
            cv2.putText(frame, "[ALERTA] EXCESSO DE PISCADAS!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

        if mar > MOUTH_AR_THRESH and not YAWN_ON:
            YAWN_ON = True
            # Thread(target=sound_alarm, daemon=True).start()
            cv2.putText(frame, "[ALERTA] BOCEJO!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        elif mar <= MOUTH_AR_THRESH:
            YAWN_ON = False

        # # textos
        # cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # cv2.putText(frame, f"BLINKS: {blinks_in_window}", (300, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()