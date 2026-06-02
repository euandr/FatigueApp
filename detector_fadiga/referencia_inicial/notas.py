# import cv2
# import numpy as np
# import mediapipe as mp
# from collections import deque

# # =========================================================
# # MEDIAPIPE
# # =========================================================
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

# # =========================================================
# # LANDMARKS MAIS ESTÁVEIS
# # =========================================================
# IDX_NOSE = 1
# IDX_LEFT_EYE = 33
# IDX_RIGHT_EYE = 263
# IDX_MOUTH_LEFT = 61
# IDX_MOUTH_RIGHT = 291
# IDX_CHIN = 152

# # =========================================================
# # MODELO 3D DA FACE
# # =========================================================
# model_points = np.array([
#     (0.0, 0.0, 0.0),          # nariz
#     (-30.0, -30.0, -30.0),   # olho esquerdo
#     (30.0, -30.0, -30.0),    # olho direito
#     (0.0, -65.0, -5.0),      # queixo
#     (-25.0, 30.0, -30.0),    # boca esquerda
#     (25.0, 30.0, -30.0)      # boca direita
# ], dtype=np.float64)

# # =========================================================
# # CONFIGURAÇÕES
# # =========================================================
# THRESH_PITCH = -70

# # suavização
# ALPHA = 0.90

# # impede saltos absurdos
# MAX_JUMP = 30

# pitch_history = deque(maxlen=10)

# smooth_pitch = 0

# # =========================================================
# # PEGA PONTOS 2D
# # =========================================================
# def get_2d_points(landmarks, w, h):

#     return np.array([
#         (
#             landmarks[IDX_NOSE].x * w,
#             landmarks[IDX_NOSE].y * h
#         ),

#         (
#             landmarks[IDX_LEFT_EYE].x * w,
#             landmarks[IDX_LEFT_EYE].y * h
#         ),

#         (
#             landmarks[IDX_RIGHT_EYE].x * w,
#             landmarks[IDX_RIGHT_EYE].y * h
#         ),

#         (
#             landmarks[IDX_CHIN].x * w,
#             landmarks[IDX_CHIN].y * h
#         ),

#         (
#             landmarks[IDX_MOUTH_LEFT].x * w,
#             landmarks[IDX_MOUTH_LEFT].y * h
#         ),

#         (
#             landmarks[IDX_MOUTH_RIGHT].x * w,
#             landmarks[IDX_MOUTH_RIGHT].y * h
#         )

#     ], dtype=np.float64)

# # =========================================================
# # CALCULA PITCH
# # =========================================================
# def get_pitch(image_points, w, h):

#     focal_length = w

#     center = (w / 2, h / 2)

#     camera_matrix = np.array([
#         [focal_length, 0, center[0]],
#         [0, focal_length, center[1]],
#         [0, 0, 1]
#     ], dtype=np.float64)

#     dist_coeffs = np.zeros((4, 1))

#     # =====================================================
#     # solvePnP mais estável
#     # =====================================================
#     success, rotation_vector, translation_vector = cv2.solvePnP(
#         model_points,
#         image_points,
#         camera_matrix,
#         dist_coeffs,
#         flags=cv2.SOLVEPNP_EPNP
#     )

#     print("solvePnP:", success)

#     if not success:
#         return None

#     # vetor → matriz rotação
#     rmat, _ = cv2.Rodrigues(rotation_vector)

#     sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)

#     singular = sy < 1e-6

#     if not singular:

#         pitch = np.arctan2(
#             rmat[2, 1],
#             rmat[2, 2]
#         )

#     else:

#         pitch = np.arctan2(
#             -rmat[1, 2],
#             rmat[1, 1]
#         )

#     pitch = np.degrees(pitch)

#     return pitch

# # =========================================================
# # WEBCAM
# # =========================================================
# cap = cv2.VideoCapture(0)

# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as face_mesh:

#     while cap.isOpened():

#         ret, frame = cap.read()

#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)

#         h, w = frame.shape[:2]

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         results = face_mesh.process(rgb)

#         if results.multi_face_landmarks:

#             face_landmarks = results.multi_face_landmarks[0]

#             landmarks = face_landmarks.landmark

#             image_points = get_2d_points(
#                 landmarks,
#                 w,
#                 h
#             )

#             pitch = get_pitch(
#                 image_points,
#                 w,
#                 h
#             )

#             if pitch is not None:

#                 # =============================================
#                 # impede saltos absurdos
#                 # =============================================
#                 if len(pitch_history) > 0:

#                     last_pitch = pitch_history[-1]

#                     if abs(pitch - last_pitch) > MAX_JUMP:
#                         pitch = last_pitch

#                 pitch_history.append(pitch)

#                 # =============================================
#                 # suavização exponencial
#                 # =============================================
#                 smooth_pitch = (
#                     ALPHA * smooth_pitch
#                     + (1 - ALPHA) * pitch
#                 )

#                 # =============================================
#                 # STATUS
#                 # =============================================
#                 if smooth_pitch > THRESH_PITCH:

#                     status = "POSTURA RUIM"
#                     color = (0, 0, 255)

#                 else:

#                     status = "POSTURA OK"
#                     color = (0, 255, 0)

#                 # =============================================
#                 # TEXTO
#                 # =============================================
#                 cv2.putText(
#                     frame,
#                     f"Pitch: {smooth_pitch:.2f}",
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     color,
#                     2
#                 )

#                 cv2.putText(
#                     frame,
#                     status,
#                     (20, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     color,
#                     2
#                 )

#             # =============================================
#             # DESENHA LANDMARKS
#             # =============================================
#             mp_drawing.draw_landmarks(
#                 frame,
#                 face_landmarks,
#                 mp_face_mesh.FACEMESH_CONTOURS
#             )

#         cv2.imshow("Head Pose Estimation", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==============================
# CONFIG
# ==============================
ALPHA = 0.95
THRESH = 0.06

smooth_score = 0

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
    scale = abs(left_sh.x - right_sh.x)

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
    if DiferencaEarAltura > 0.05:  
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
                # SUAVIZAÇÃO (EMA)
                # ==========================
                smooth_score = ALPHA * smooth_score + (1 - ALPHA) * score

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