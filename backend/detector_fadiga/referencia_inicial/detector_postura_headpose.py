import sys
import os
import time
import cv2
import numpy as np

# adiciona a pasta deep-head-pose/code ao path para importar utils
HERE = os.path.dirname(__file__)
DEEP_HP = os.path.normpath(os.path.join(HERE, '..', 'deep-head-pose', 'code'))
if DEEP_HP not in sys.path:
    sys.path.insert(0, DEEP_HP)

try:
    import utils as dhp_utils
except Exception:
    dhp_utils = None

import mediapipe as mp
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# parâmetros simples
THRESH = 0.06
ALPHA = 0.90
ALPHA_FAST = 0.6
SMOOTH_ADAPT_THRESHOLD = 0.08

smooth_score = 0.0
smooth_initialized = False

# reutiliza lógica de score (normalize por ombros/torso)
def posture_score(lm):
    nose = lm[mp_holistic.PoseLandmark.NOSE]
    left_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]

    shoulder_y = (left_sh.y + right_sh.y) / 2
    shoulder_z = (left_sh.z + right_sh.z) / 2

    shoulder_width = abs(left_sh.x - right_sh.x)
    left_hip = lm[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = lm[mp_holistic.PoseLandmark.RIGHT_HIP]
    torso_height = abs(((left_sh.y + right_sh.y) / 2) - ((left_hip.y + right_hip.y) / 2))
    scale = max(shoulder_width, torso_height)
    if scale < 1e-6:
        return None
    pitch_y = (nose.y - shoulder_y) / scale
    depth_z = (shoulder_z - nose.z) / scale
    score = pitch_y + 0.5 * depth_z
    return score

# inclinação da cabeça por diferença entre orelhas/eyes
def Inclinacao_cabeca(landmarks):
    left_ear = landmarks[mp_holistic.PoseLandmark.LEFT_EAR]
    right_ear = landmarks[mp_holistic.PoseLandmark.RIGHT_EAR]
    diff = abs(left_ear.y - right_ear.y)
    return {"diferenca": diff, "inclinacao": diff > 0.07}


def approx_head_angles(lm, img_w, img_h):
    """Estima yaw/pitch/roll a partir de landmarks (aproximação simples).
    Retorna (yaw_deg, pitch_deg, roll_deg, tdx, tdy, scale)
    """
    nose = lm[mp_holistic.PoseLandmark.NOSE]
    left_sh = lm[mp_holistic.PoseLandmark.LEFT_SHOULDER]
    right_sh = lm[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
    left_eye = lm[mp_holistic.PoseLandmark.LEFT_EYE]
    right_eye = lm[mp_holistic.PoseLandmark.RIGHT_EYE]

    mid_sh_x = (left_sh.x + right_sh.x) / 2.0
    mid_sh_y = (left_sh.y + right_sh.y) / 2.0

    # escala baseada em ombros/torso (similar ao posture_score)
    left_hip = lm[mp_holistic.PoseLandmark.LEFT_HIP]
    right_hip = lm[mp_holistic.PoseLandmark.RIGHT_HIP]
    shoulder_width = abs(left_sh.x - right_sh.x)
    torso_height = abs(((left_sh.y + right_sh.y) / 2) - ((left_hip.y + right_hip.y) / 2))
    scale = max(shoulder_width, torso_height, 1e-6)

    # yaw: quanto o nariz se desloca em x relativo ao centro dos ombros
    yaw = (nose.x - mid_sh_x) / scale
    # pitch: quanto o nariz sobe/ desce relativo aos ombros
    pitch = (nose.y - mid_sh_y) / scale
    # roll: diferença vertical entre olhos (aprox.)
    roll = (left_eye.y - right_eye.y) * 5.0

    # converte para graus com fator heurístico
    yaw_deg = yaw * 60.0
    pitch_deg = pitch * 60.0
    roll_deg = roll * 60.0

    tdx = int(nose.x * img_w)
    tdy = int(nose.y * img_h)

    return yaw_deg, pitch_deg, roll_deg, tdx, tdy, scale


def main():
    global smooth_score, smooth_initialized
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # flip horizontal para espelhar (como usado em outras rotinas)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
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

                    # estimativa de ângulos
                    yaw_deg, pitch_deg, roll_deg, tdx, tdy, scale = approx_head_angles(lm, w, h)

                    # desenha eixos usando deep-head-pose utils se disponível
                    if dhp_utils is not None:
                        try:
                            dhp_utils.draw_axis(frame, yaw_deg, pitch_deg, roll_deg, tdx=tdx, tdy=tdy, size=int(scale * min(w,h) * 0.5))
                        except Exception:
                            pass

                    if inclinacao_info["inclinacao"] or smooth_score > THRESH:
                        status = "postura ruim"
                        color = (0, 0, 255)
                    else:
                        status = "POSTURA OK"
                        color = (0, 255, 0)

                    # cv2.putText(frame, f"ScoreFrente: {smooth_score:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    # # cv2.putText(frame, f"ScoreBruto: {score:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    # cv2.putText(frame, f"ScoreLados: {inclinacao_info['diferenca']:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    # cv2.putText(frame, status, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # desenha landmarks
            try:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            except Exception:
                pass

            cv2.imshow('Detector de Postura - headpose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
