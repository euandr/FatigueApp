import cv2
import mediapipe as mp
import imutils
import time

# Inicializa MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Captura vídeo webcam
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa frame e detecta landmarks
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w = frame.shape[:2]
        for face_landmarks in results.multi_face_landmarks:
            # Desenha cada ponto como um círculo vermelho (raio 1)
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
