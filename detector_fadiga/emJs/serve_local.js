#!/usr/bin/env node
/**
 * Servidor WebSocket para Detecção de Fadiga
 * ------------------------------------------
 * Execute este servidor localmente para processar os frames da webcam.
 *
 * Instalação das dependências:
 *     npm install ws @mediapipe/tasks-vision sharp dotenv pino
 *
 * O servidor irá escutar em:
 *     - Local: ws://localhost:8765 (ou porta configurada)
 *     - Remoto: ws://seu-ip:8765 (quando WS_HOST=0.0.0.0, padrão)
 */

const WebSocket = require("ws");
const sharp = require("sharp");
const { FaceLandmarker, FilesetResolver } = require("@mediapipe/tasks-vision");
const { Buffer } = require("buffer");
require("dotenv").config();
const pino = require("pino");

// Configurar logging
const logger = pino();

// Constantes de detecção
const EYE_AR_THRESH = 0.2;
const EYE_AR_CONSEC_FRAMES = 30;
const BLINK_THRESH = 0.2;
const BLINK_CONSEC_FRAMES = 3;
const MOUTH_AR_THRESH = 0.6;
const EXCESS_BLINKS_THRESH = 5; // 5 piscadas na janela de 30 frames para alerta

// Índices dos landmarks do MediaPipe Face Mesh
const LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380];
const MOUTH_IDX = [13, 14, 78, 308]; // 13,14 vertical; 78,308 horizontal

// MediaPipe setup
let faceLandmarker;

/**
 * Inicializa o FaceLandmarker do MediaPipe
 */
async function initializeFaceLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-studio/latest/face_landmarker.task`,
      },
      runningMode: "IMAGE",
      numFaces: 1,
    });
    logger.info("MediaPipe FaceLandmarker inicializado com sucesso");
  } catch (error) {
    logger.error({ error }, "Erro ao inicializar FaceLandmarker");
    throw error;
  }
}

/**
 * Calcula a distância euclidiana entre dois pontos
 */
function euclideanDistance(p1, p2) {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z || 0) - (p2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calcula o Eye Aspect Ratio (EAR)
 */
function eyeAspectRatio(eye) {
  const A = euclideanDistance(eye[1], eye[5]);
  const B = euclideanDistance(eye[2], eye[4]);
  const C = euclideanDistance(eye[0], eye[3]);
  return C > 0 ? (A + B) / (2.0 * C) : 0;
}

/**
 * Calcula o Mouth Aspect Ratio (MAR)
 */
function mouthAspectRatio(mouth) {
  const A = euclideanDistance(mouth[0], mouth[1]); // vertical
  const B = euclideanDistance(mouth[2], mouth[3]); // horizontal
  return B > 0 ? A / B : 0;
}

// Variáveis globais para estado da detecção
let earHistory = [];
let counter = 0;
let blinkCounter = 0;
let totalBlinks = 0;
let alarmOn = false;
let yawnOn = false;
let blinkAlarmOn = false;

/**
 * Processa um frame e retorna dados de detecção
 */
async function processFrame(frameBuffer) {
  let detectionData = {
    ear: 0.3,
    mar: 0.2,
    blinks: 0,
    totalBlinks: totalBlinks,
    eyesClosed: false,
    yawnDetected: false,
    excessBlinks: false,
    fatigueAlert: false,
  };

  try {
    // Processar landmarks com MediaPipe
    const results = faceLandmarker.detectForVideo(frameBuffer, Date.now());

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];

      // Extrair landmarks dos olhos e boca
      const leftEye = LEFT_EYE_IDX.map((idx) => landmarks[idx]);
      const rightEye = RIGHT_EYE_IDX.map((idx) => landmarks[idx]);
      const mouth = MOUTH_IDX.map((idx) => landmarks[idx]);

      // Calcular métricas
      const leftEar = eyeAspectRatio(leftEye);
      const rightEar = eyeAspectRatio(rightEye);
      const ear = (leftEar + rightEar) / 2.0;
      const mar = mouthAspectRatio(mouth);

      // Histórico de EAR
      earHistory.push(ear);
      if (earHistory.length > 30) {
        earHistory.shift();
      }

      // Contar piscadas na janela
      let blinksInWindow = 0;
      for (let i = 1; i < earHistory.length; i++) {
        if (
          earHistory[i - 1] > EYE_AR_THRESH &&
          earHistory[i] < EYE_AR_THRESH
        ) {
          blinksInWindow += 1;
        }
      }

      // Detecção de fadiga (olhos fechados por tempo)
      const eyesClosed = ear < EYE_AR_THRESH;
      let fatigueAlert = false;

      if (eyesClosed) {
        counter += 1;
        blinkCounter += 1;

        if (counter >= EYE_AR_CONSEC_FRAMES) {
          fatigueAlert = true;
          if (!alarmOn) {
            alarmOn = true;
          }
        }

        // Detecção de piscada
        if (blinkCounter >= BLINK_CONSEC_FRAMES) {
          totalBlinks += 1;
          blinkCounter = 0;
        }
      } else {
        counter = 0;
        alarmOn = false;
        blinkCounter = 0;
      }

      // Alarme para excesso de piscadas
      const excessBlinks = blinksInWindow >= EXCESS_BLINKS_THRESH;
      if (excessBlinks) {
        if (!blinkAlarmOn) {
          blinkAlarmOn = true;
        }
      } else {
        blinkAlarmOn = false;
      }

      // Detecção de bocejo
      const yawnDetected = mar > MOUTH_AR_THRESH;
      if (yawnDetected && !yawnOn) {
        yawnOn = true;
      } else if (!yawnDetected) {
        yawnOn = false;
      }

      // Preparar dados de detecção
      detectionData = {
        ear: parseFloat(ear.toFixed(6)),
        mar: parseFloat(mar.toFixed(6)),
        blinks: blinksInWindow,
        totalBlinks: totalBlinks,
        yawnDetected: yawnDetected && yawnOn,
        excessBlinks: excessBlinks,
        fatigueAlert: fatigueAlert,
      };
    }
  } catch (error) {
    logger.error({ error }, "Erro ao processar frame");
  }

  // Codificar frame para base64
  try {
    const frameBase64 = await sharp(frameBuffer)
      .jpeg({ quality: 70 })
      .toBuffer()
      .then((buffer) => buffer.toString("base64"));

    detectionData.frame = frameBase64;
  } catch (error) {
    logger.error({ error }, "Erro ao codificar frame");
  }

  return detectionData;
}

/**
 * Handler para conexões WebSocket
 */
async function handleClient(ws, req) {
  const clientAddr = req.socket.remoteAddress;
  logger.info(`Cliente conectado: ${clientAddr}`);

  ws.on("message", async (message) => {
    try {
      // Verificar se é um buffer ou string
      let data;
      if (typeof message === "string") {
        data = JSON.parse(message);
      } else if (Buffer.isBuffer(message)) {
        data = JSON.parse(message.toString());
      } else {
        data = JSON.parse(String(message));
      }

      if (data.frame) {
        // Decodificar frame base64
        const frameBuffer = Buffer.from(data.frame, "base64");

        // Processar frame
        const result = await processFrame(frameBuffer);

        // Enviar resultado
        ws.send(JSON.stringify(result));
      }
    } catch (error) {
      if (error instanceof SyntaxError) {
        logger.error(`Erro ao decodificar JSON: ${error.message}`);
      } else {
        logger.error({ error }, "Erro ao processar frame");
      }
    }
  });

  ws.on("close", () => {
    logger.info(`Cliente desconectado: ${clientAddr}`);
  });

  ws.on("error", (error) => {
    logger.error({ error }, "Erro na conexão WebSocket");
  });
}

/**
 * Inicia o servidor WebSocket
 */
async function main() {
  // Inicializar MediaPipe
  await initializeFaceLandmarker();

  // Para produção: use "0.0.0.0" para aceitar conexões externas
  // Para desenvolvimento local: use "localhost"
  const host = process.env.WS_HOST || "0.0.0.0"; // Aceita conexões de qualquer IP
  const port = parseInt(process.env.WS_PORT || "8765");

  const wss = new WebSocket.Server({ host, port });

  logger.info(
    `Iniciando servidor de detecção de fadiga em ws://${host}:${port}`,
  );
  logger.info("Pressione Ctrl+C para encerrar");

  wss.on("connection", handleClient);

  process.on("SIGINT", () => {
    logger.info("Servidor encerrado");
    wss.close(() => {
      process.exit(0);
    });
  });
}

// Executar servidor
main().catch((error) => {
  logger.error({ error }, "Erro ao iniciar servidor");
  process.exit(1);
});

/*
 * Deu certo com esse
 * https://lovable.dev/projects/1e7fc945-aa49-4f21-9ac1-1517fb7a130f
 * na conta teste@gmail.com
 */
