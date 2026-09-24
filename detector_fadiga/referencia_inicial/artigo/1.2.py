import cv2
import numpy as np
import mediapipe as mp

# =========================
# MediaPipe Face Mesh
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# Landmarks utilizados
# =========================

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]

# =========================
# Webcam
# =========================

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        mesh_points = results.multi_face_landmarks[0].landmark

        h, w = frame.shape[:2]

        left_eye = np.array([
            (int(mesh_points[i].x * w),
             int(mesh_points[i].y * h))
            for i in LEFT_EYE_IDX
        ])

        right_eye = np.array([
            (int(mesh_points[i].x * w),
             int(mesh_points[i].y * h))
            for i in RIGHT_EYE_IDX
        ])

        mouth = np.array([
            (int(mesh_points[i].x * w),
             int(mesh_points[i].y * h))
            for i in MOUTH_IDX
        ])

        # =========================
        # Malha facial completa
        # =========================

        for lm in mesh_points:
            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(
                frame,
                (x, y),
                1,
                (180, 180, 180),
                -1
            )

        # =========================
        # Olhos (EAR)
        # =========================

        for (x, y) in left_eye:
            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        for (x, y) in right_eye:
            cv2.circle(
                frame,
                (x, y),
                3,
                (0, 255, 0),
                -1
            )

        # cv2.polylines(
        #     frame,
        #     [left_eye],
        #     True,
        #     (0, 255, 0),
        #     1
        # )

        # cv2.polylines(
        #     frame,
        #     [right_eye],
        #     True,
        #     (0, 255, 0),
        #     1
        # )

        # =========================
        # Boca (MAR)
        # =========================

        for (x, y) in mouth:
            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

        # cv2.line(
        #     frame,
        #     tuple(mouth[0]),
        #     tuple(mouth[1]),
        #     (255, 0, 0),
        #     2
        # )

        # cv2.line(
        #     frame,
        #     tuple(mouth[2]),
        #     tuple(mouth[3]),
        #     (255, 0, 0),
        #     2
        # )

        # =========================
        # Legenda
        # =========================

        cv2.putText(
            frame,
            "Landmarks dos Olhos (EAR)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Landmarks da Boca (MAR)",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # cv2.putText(
        #     frame,
        #     "Pressione S para salvar a figura",
        #     (20, 120),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (255, 255, 255),
        #     2
        # )

    cv2.imshow(
        "Landmarks utilizados no sistema",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite(
            "landmarks_olhos_boca.png",
            frame
        )
        print("Imagem salva!")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()