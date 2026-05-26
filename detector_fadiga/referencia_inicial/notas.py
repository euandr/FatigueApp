import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# --- Histórico para suavização ---
FORWARD_HISTORY = []
FORWARD_HEAD_THRESH = 1.78

def smooth_offset(offset, history, window=30):
    history.append(offset)
    if len(history) > window:
        history.pop(0)
    return np.mean(history)

def Cabeca_projetada_para_frente(landmarks):
    nose     = landmarks[mp_holistic.PoseLandmark.NOSE]
    left_sh  = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_sh = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    shoulder_mid_z = (left_sh.z + right_sh.z) / 2.0

    # para divisao por 0 não acontecer
    if abs(shoulder_mid_z) < 0.001:
        return None

    # abs(-5 )= 5, abs(-0.0001)= 0.0001 
    diff = (shoulder_mid_z - nose.z) / abs(shoulder_mid_z) # normalizar a diferença para funcionar independente da distância da câmera.

    return diff


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is None:
            print("No pose landmarks detected.")
        else:
            landmarks = results.pose_landmarks.landmark

            # 1. Cabeça projetada para frente
            offset = Cabeca_projetada_para_frente(landmarks)

            if offset is not None:
                smooth = smooth_offset(offset, FORWARD_HISTORY)

                # HUD
                cv2.putText(image, f"FWD: {smooth:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if smooth > FORWARD_HEAD_THRESH:
                    cv2.putText(image, "[ALERTA] CABECA PROJETADA", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(image, "POSTURA OK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0),    thickness=2, circle_radius=2))

        cv2.namedWindow('Raw Webcam Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Raw Webcam Feed', 1100, 800)
        cv2.imshow('Raw Webcam Feed',image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()