#!/usr/bin/env python3
"""
Servidor WebSocket para Detecção de Fadiga
------------------------------------------
Execute este servidor localmente para processar os frames da webcam.

Instalação das dependências:
    pip install websockets opencv-python mediapipe scipy numpy

EXECUÇÃO LOCAL:
    Opção 1 - Executar diretamente (aceita conexões locais e remotas):
        python serve.py
        
    Opção 2 - Forçar apenas conexões locais (localhost):
        Windows PowerShell:
            $env:WS_HOST="localhost"
            python serve.py
        
        Windows CMD:
            set WS_HOST=localhost
            python serve.py
        
        Linux/Mac:
            export WS_HOST=localhost
            python serve.py
    
    Opção 3 - Personalizar porta:
        Windows PowerShell:
            $env:WS_PORT="9000"
            python serve.py
        
        Linux/Mac:
            export WS_PORT=9000
            python serve.py

O servidor irá escutar em:
    - Local: ws://localhost:8765 (ou porta configurada)
    - Remoto: ws://seu-ip:8765 (quando WS_HOST=0.0.0.0, padrão)

Para parar o servidor: Pressione Ctrl+C
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
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes de detecção
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
BLINK_THRESH = 0.2
BLINK_CONSEC_FRAMES = 3
MOUTH_AR_THRESH = 0.6
EXCESS_BLINKS_THRESH = 8  # Aumentado: 8 piscadas na janela de 30 frames para alerta

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


# Variáveis globais para estado da detecção
ear_history = []
counter = 0
blink_counter = 0
total_blinks = 0
alarm_on = False
yawn_on = False
blink_alarm_on = False


def process_frame(frame):
    """Processa um frame e retorna dados de detecção"""
    global ear_history, counter, blink_counter, total_blinks
    global alarm_on, yawn_on, blink_alarm_on
    
    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    detection_data = {
        "ear": 0.3,
        "mar": 0.2,
        "blinks": 0,
        "totalBlinks": total_blinks,
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
        ear_history.append(ear)
        if len(ear_history) > 30:
            ear_history.pop(0)

        # Contar piscadas na janela
        blinks_in_window = 0
        for i in range(1, len(ear_history)):
            if ear_history[i-1] > EYE_AR_THRESH and ear_history[i] < EYE_AR_THRESH:
                blinks_in_window += 1

        # Detecção de fadiga (olhos fechados por tempo)
        eyes_closed = ear < EYE_AR_THRESH
        fatigue_alert = False
        
        if eyes_closed:
            counter += 1
            blink_counter += 1
            
            if counter >= EYE_AR_CONSEC_FRAMES:
                fatigue_alert = True
                if not alarm_on:
                    alarm_on = True
            
            # Detecção de piscada
            if blink_counter >= BLINK_CONSEC_FRAMES:
                total_blinks += 1
                blink_counter = 0
        else:
            counter = 0
            alarm_on = False
            blink_counter = 0

        # Alarme para excesso de piscadas
        excess_blinks = blinks_in_window >= EXCESS_BLINKS_THRESH
        if excess_blinks:
            if not blink_alarm_on:
                blink_alarm_on = True
        else:
            blink_alarm_on = False

        # Detecção de bocejo
        yawn_detected = mar > MOUTH_AR_THRESH
        if yawn_detected and not yawn_on:
            yawn_on = True
        elif not yawn_detected:
            yawn_on = False

        # Preparar dados de detecção
        detection_data = {
            "ear": float(ear),
            "mar": float(mar),
            "blinks": blinks_in_window,
            "totalBlinks": total_blinks,
            "eyesClosed": eyes_closed,
            "yawnDetected": yawn_detected and yawn_on,
            "excessBlinks": excess_blinks,
            "fatigueAlert": fatigue_alert,
        }

    # Codificar frame para base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    detection_data["frame"] = frame_base64
    # Converter valores numpy para tipos nativos Python
    detection_data = {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in detection_data.items()}

    return detection_data


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
                        result = process_frame(frame)
                        
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
    # Para produção: use "0.0.0.0" para aceitar conexões externas
    # Para desenvolvimento local: use "localhost"
    host = os.getenv("WS_HOST", "0.0.0.0")  # Aceita conexões de qualquer IP
    port = int(os.getenv("WS_PORT", "8765"))
    
    logger.info(f"Iniciando servidor de detecção de fadiga em ws://{host}:{port}")
    logger.info("Pressione Ctrl+C para encerrar")
    
    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # Rodar para sempre


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Servidor encerrado")



"""
 deu certo com esse
 https://lovable.dev/projects/1e7fc945-aa49-4f21-9ac1-1517fb7a130f
 na conta teste@gmail.com
"""