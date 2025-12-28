#!/usr/bin/env python3
"""
Servidor WebSocket para Detecção de Fadiga
------------------------------------------
Execute este servidor localmente para processar os frames da webcam.

Instalação das dependências:
    pip install websockets opencv-python mediapipe scipy numpy

Execução:
    python fatigue_server.py

O servidor irá escutar em ws://localhost:8765
"""

import asyncio
import websockets
import json
import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import base64
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes de detecção
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
BLINK_THRESH = 0.2
BLINK_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.6

# Índices dos landmarks do MediaPipe Face Mesh
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [13, 14, 78, 308]  # 13,14 vertical; 78,308 horizontal

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def eye_aspect_ratio(eye):
    """Calcula o Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C > 0 else 0


def mouth_aspect_ratio(mouth):
    """Calcula o Mouth Aspect Ratio (MAR)"""
    A = dist.euclidean(mouth[0], mouth[1])  # vertical
    B = dist.euclidean(mouth[2], mouth[3])  # horizontal
    return A / B if B > 0 else 0


class FatigueDetector:
    def __init__(self):
        self.ear_history = []
        self.counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        self.alarm_on = False
        self.yawn_on = False
        self.blink_alarm_on = False

    def process_frame(self, frame):
        """Processa um frame e retorna dados de detecção"""
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        detection_data = {
            "ear": 0.3,
            "mar": 0.2,
            "blinks": 0,
            "totalBlinks": self.total_blinks,
            "eyesClosed": False,
            "yawnDetected": False,
            "excessBlinks": False,
            "fatigueAlert": False,
        }

        if results.multi_face_landmarks:
            mesh_points = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            # Extrair landmarks
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

            # Calcular métricas
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Histórico de EAR
            self.ear_history.append(ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)

            # Desenhar landmarks
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            cv2.line(frame, tuple(mouth[0]), tuple(mouth[1]), (255, 0, 0), 2)
            cv2.line(frame, tuple(mouth[2]), tuple(mouth[3]), (255, 0, 0), 2)

            # Contar piscadas na janela
            blinks_in_window = 0
            for i in range(1, len(self.ear_history)):
                if self.ear_history[i-1] > EYE_AR_THRESH and self.ear_history[i] < EYE_AR_THRESH:
                    blinks_in_window += 1

            # Detecção de fadiga (olhos fechados por tempo)
            eyes_closed = ear < EYE_AR_THRESH
            fatigue_alert = False
            
            if eyes_closed:
                self.counter += 1
                self.blink_counter += 1
                
                if self.counter >= EYE_AR_CONSEC_FRAMES:
                    fatigue_alert = True
                    if not self.alarm_on:
                        self.alarm_on = True
                    cv2.putText(frame, "[ALERTA] FADIGA!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Detecção de piscada
                if self.blink_counter >= BLINK_CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.blink_counter = 0
            else:
                self.counter = 0
                self.alarm_on = False
                self.blink_counter = 0

            # Alarme para excesso de piscadas
            excess_blinks = blinks_in_window >= 5
            if excess_blinks:
                if not self.blink_alarm_on:
                    self.blink_alarm_on = True
                cv2.putText(frame, "[ALERTA] EXCESSO DE PISCADAS!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
            else:
                self.blink_alarm_on = False

            # Detecção de bocejo
            yawn_detected = mar > MOUTH_AR_THRESH
            if yawn_detected and not self.yawn_on:
                self.yawn_on = True
                cv2.putText(frame, "[ALERTA] BOCEJO!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif not yawn_detected:
                self.yawn_on = False

            # Mostrar métricas no frame
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.3f}", (10, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blinks_in_window}", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Preparar dados de detecção
            detection_data = {
                "ear": float(ear),
                "mar": float(mar),
                "blinks": blinks_in_window,
                "totalBlinks": self.total_blinks,
                "eyesClosed": eyes_closed,
                "yawnDetected": yawn_detected and self.yawn_on,
                "excessBlinks": excess_blinks,
                "fatigueAlert": fatigue_alert,
            }

        # Codificar frame para base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        detection_data["frame"] = frame_base64

        return detection_data


# Instância global do detector
detector = FatigueDetector()


async def handle_client(websocket):
    """Handler para conexões WebSocket"""
    client_addr = websocket.remote_address
    logger.info(f"Cliente conectado: {client_addr}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                if "frame" in data:
                    # Decodificar frame base64
                    frame_data = base64.b64decode(data["frame"])
                    np_arr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Processar frame
                        result = detector.process_frame(frame)
                        
                        # Enviar resultado
                        await websocket.send(json.dumps(result))
                    else:
                        logger.warning("Frame inválido recebido")
                        
            except json.JSONDecodeError as e:
                logger.error(f"Erro ao decodificar JSON: {e}")
            except Exception as e:
                logger.error(f"Erro ao processar frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Cliente desconectado: {client_addr}")
    except Exception as e:
        logger.error(f"Erro na conexão: {e}")


async def main():
    """Inicia o servidor WebSocket"""
    host = "localhost"
    port = 8765
    
    logger.info(f"Iniciando servidor de detecção de fadiga em ws://{host}:{port}")
    logger.info("Pressione Ctrl+C para encerrar")
    
    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # Rodar para sempre


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servidor encerrado")
