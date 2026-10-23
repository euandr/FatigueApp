"""Algoritmo de serve_local.py, com o estado isolado por câmera/sessão.

Os limiares, cálculos, condições e campos de saída do protótipo são preservados.
"""
import base64

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
BLINK_THRESH = 0.2
BLINK_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.6
EXCESS_BLINKS_THRESH = 5
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0


def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[0], mouth[1])
    B = dist.euclidean(mouth[2], mouth[3])
    return A / B if B > 0 else 0


class Detector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.ear_history = []
        self.counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        self.alarm_on = False
        self.yawn_on = False
        self.blink_alarm_on = False

    def close(self):
        self.face_mesh.close()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        detection_data = {
            "ear": 0.3, "mar": 0.2, "blinks": 0,
            "totalBlinks": self.total_blinks, "eyesClosed": False,
            "yawnDetected": False, "excessBlinks": False, "fatigueAlert": False,
        }
        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left_eye = np.array([
                (int(mesh_points[i].x * w), int(mesh_points[i].y * h))
                for i in LEFT_EYE_IDX
            ])
            right_eye = np.array([
                (int(mesh_points[i].x * w), int(mesh_points[i].y * h))
                for i in RIGHT_EYE_IDX
            ])
            mouth = np.array([
                (int(mesh_points[i].x * w), int(mesh_points[i].y * h))
                for i in MOUTH_IDX
            ])
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)
            self.ear_history.append(ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)
            blinks_in_window = 0
            for i in range(1, len(self.ear_history)):
                if self.ear_history[i-1] > EYE_AR_THRESH and self.ear_history[i] < EYE_AR_THRESH:
                    blinks_in_window += 1
            eyes_closed = ear < EYE_AR_THRESH
            fatigue_alert = False
            if eyes_closed:
                self.counter += 1
                self.blink_counter += 1
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    fatigue_alert = True
                    if not self.alarm_on:
                        self.alarm_on = True
                if self.blink_counter >= BLINK_CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.blink_counter = 0
            else:
                self.counter = 0
                self.alarm_on = False
                self.blink_counter = 0
            excess_blinks = blinks_in_window >= EXCESS_BLINKS_THRESH
            if excess_blinks:
                if not self.blink_alarm_on:
                    self.blink_alarm_on = True
            else:
                self.blink_alarm_on = False
            yawn_detected = mar > MOUTH_AR_THRESH
            if yawn_detected and not self.yawn_on:
                self.yawn_on = True
            elif not yawn_detected:
                self.yawn_on = False
            detection_data = {
                "ear": float(ear), "mar": float(mar), "blinks": blinks_in_window,
                "totalBlinks": self.total_blinks,
                "yawnDetected": yawn_detected and self.yawn_on,
                "excessBlinks": excess_blinks, "fatigueAlert": fatigue_alert,
            }
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        detection_data["frame"] = base64.b64encode(buffer).decode('utf-8')
        return {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in detection_data.items()}
