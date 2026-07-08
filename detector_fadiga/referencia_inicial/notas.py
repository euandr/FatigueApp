

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










cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.pose_landmarks:

            lm = results.pose_landmarks.landmark

            score = posture_score(lm)

            if score is not None:

                # ==========================
                # SUAVIZAÇÃO (EMA adaptativa)
                # - inicializa na primeira amostra
                # - usa ganho maior (ALPHA_FAST) quando houver grande mudança
                # ==========================
                if not smooth_initialized:
                    smooth_score = score
                    smooth_initialized = True
                else:
                    diff = abs(score - smooth_score)
                    alpha = ALPHA_FAST if diff > SMOOTH_ADAPT_THRESHOLD else ALPHA
                    smooth_score = alpha * smooth_score + (1 - alpha) * score

                # verificar inclinação lateral
                inclinacao_info = Inclinacao_cabeca(lm)

                # ==========================
                # DECISÃO
                # ==========================
                if inclinacao_info["inclinacao"]:
                    status = "postura ruim"
                    color = (0, 0, 255)
                elif smooth_score > THRESH:
                    status = "postura ruim"
                    color = (0, 0, 255)
                else:
                    status = "POSTURA OK"
                    color = (0, 255, 0)

                # ==========================
                # HUD
                # ==========================
                cv2.putText(
                    frame,
                    f"ScoreFrente: {smooth_score:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # mostrar score bruto para comparação (ajuste/depuração)
                cv2.putText(
                    frame,
                    f"ScoreBruto: {score:.3f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    1
                )
                cv2.putText(
                    frame,
                    f"ScoreLados: {inclinacao_info['diferenca']:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        cv2.imshow("Postura - Y + Z Fusion", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()