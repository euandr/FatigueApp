import { useState, useRef, useCallback, useEffect } from "react";
import alarmSound from "@/assets/alarm-clock.mp3";
import { WS_URL } from "@/services/api";

const initialMetrics = {
  ear: 0.3,
  mar: 0.2,
  blinks: 0,
  totalBlinks: 0,
  eyesClosed: false,
  yawnDetected: false,
  excessBlinks: false,
  fatigueAlert: false,
};

/**
 * Detecção local: captura a webcam na própria tela e envia os frames para o
 * backend via WebSocket. Não autentica, não cria sessão e não grava nada.
 */
export const useFatigueDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [processedFrame, setProcessedFrame] = useState(null);
  const [events, setEvents] = useState([]);
  const [yawnCount, setYawnCount] = useState(0);
  const [metrics, setMetrics] = useState(initialMetrics);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const audioRef = useRef(null);
  const lastAlertRef = useRef({});
  const yawnActiveRef = useRef(false);
  const yawnResetIntervalRef = useRef(null);
  const isMutedRef = useRef(false);
  const reconnectTimeoutRef = useRef(null);
  const reconnectDelayRef = useRef(1000);
  const streamingRef = useRef(false);
  const startingRef = useRef(false);
  const readyRef = useRef(false);
  const pendingFrameRef = useRef(false);
  const connectRef = useRef(null);
  const generationRef = useRef(0);

  useEffect(() => {
    audioRef.current = new Audio(alarmSound);
    return () => {
      audioRef.current = null;
    };
  }, []);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const addEvent = useCallback((message, type) => {
    const now = Date.now();
    if (lastAlertRef.current[type] && now - lastAlertRef.current[type] < 2000) {
      return;
    }
    lastAlertRef.current[type] = now;
    setEvents((prev) =>
      [
        {
          id: `${now}-${Math.random().toString(36).slice(2, 11)}`,
          type,
          camera: "camera1",
          timestamp: new Date(),
          message,
        },
        ...prev,
      ].slice(0, 50)
    );
    if (!isMutedRef.current && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {});
    }
  }, []);

  const stopStreaming = useCallback(() => {
    // Invalidar callbacks antigos antes de fechar o socket evita reconexão após parar.
    streamingRef.current = false;
    generationRef.current += 1;
    startingRef.current = false;
    readyRef.current = false;
    pendingFrameRef.current = false;
    clearTimeout(reconnectTimeoutRef.current);
    clearInterval(intervalRef.current);
    clearInterval(yawnResetIntervalRef.current);
    reconnectDelayRef.current = 1000;
    yawnActiveRef.current = false;
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsStreaming(false);
    setIsConnected(false);
    setProcessedFrame(null);
    setYawnCount(0);
    setMetrics({ ...initialMetrics, ear: 0, mar: 0 });
  }, []);

  const connectWebSocket = useCallback(() => {
    if (!streamingRef.current) return;
    const generation = generationRef.current;
    const scheduleReconnect = () => {
      if (!streamingRef.current || generation !== generationRef.current) {
        return;
      }
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = setTimeout(
        () => connectRef.current(),
        reconnectDelayRef.current
      );
      reconnectDelayRef.current = Math.min(reconnectDelayRef.current * 1.5, 10000);
    };

    try {
      const socket = new WebSocket(WS_URL);
      wsRef.current = socket;
      socket.onopen = () => {
        if (generation !== generationRef.current) {
          socket.close();
        }
        // O backend envia {"type":"ready"} assim que aceita a conexão.
      };
      socket.onclose = (event) => {
        if (generation !== generationRef.current) return;
        readyRef.current = false;
        pendingFrameRef.current = false;
        setIsConnected(false);
        if (event.code === 1000 || event.code === 1008) {
          // Encerramento normal ou conexão recusada: não repetir indefinidamente.
          stopStreaming();
        } else {
          scheduleReconnect();
        }
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event) => {
        if (generation !== generationRef.current) return;
        try {
          const data = JSON.parse(event.data);
          if (data.type === "ready") {
            readyRef.current = true;
            setIsConnected(true);
            reconnectDelayRef.current = 1000;
            return;
          }
          if (data.type === "error") {
            // Libera o próximo frame mesmo quando o backend recusa um frame.
            pendingFrameRef.current = false;
            console.error("Erro da API:", data.message);
            return;
          }
          if (data.type === "pong") return;

          pendingFrameRef.current = false;
          const { frame, ...values } = data;
          setMetrics({
            ...values,
            // O detector não devolve eyesClosed quando encontra um rosto.
            eyesClosed: values.eyesClosed ?? values.ear < 0.2,
          });
          if (frame) setProcessedFrame(frame);
          if (data.fatigueAlert) {
            addEvent(
              "Alerta de fadiga detectado - olhos fechados por muito tempo",
              "fatigue_alert"
            );
          }
          // Conta cada bocejo uma vez, não um por frame enquanto a boca abre.
          if (data.yawnDetected && !yawnActiveRef.current) {
            setYawnCount((prev) => prev + 1);
            addEvent("Bocejo detectado", "yawn");
          }
          yawnActiveRef.current = Boolean(data.yawnDetected);
          if (data.excessBlinks) {
            addEvent(
              `Excesso de piscadas detectado (${data.blinks} em 30 frames)`,
              "excess_blinks"
            );
          }
        } catch (error) {
          console.error("Mensagem WebSocket inválida:", error);
        }
      };
    } catch (error) {
      console.error("Erro ao conectar WebSocket:", error);
      scheduleReconnect();
    }
  }, [addEvent, stopStreaming]);

  useEffect(() => {
    connectRef.current = connectWebSocket;
  }, [connectWebSocket]);

  const startStreaming = useCallback(async () => {
    if (streamingRef.current || startingRef.current) return;
    startingRef.current = true;
    const generation = ++generationRef.current;
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = 640;
      canvasRef.current.height = 480;
      lastAlertRef.current = {};
      streamingRef.current = true;
      setIsStreaming(true);
      connectRef.current();

      intervalRef.current = window.setInterval(() => {
        if (
          !readyRef.current ||
          pendingFrameRef.current ||
          wsRef.current?.readyState !== WebSocket.OPEN ||
          videoRef.current?.readyState < 2
        ) {
          return;
        }
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx && videoRef.current) {
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          const frame = canvasRef.current
            .toDataURL("image/jpeg", 0.7)
            .split(",")[1];
          pendingFrameRef.current = true;
          wsRef.current.send(JSON.stringify({ frame }));
        }
      }, 100);
      yawnResetIntervalRef.current = window.setInterval(
        () => setYawnCount(0),
        60000
      );
    } catch (error) {
      stream?.getTracks().forEach((track) => track.stop());
      console.error("Erro ao iniciar transmissão:", error);
      if (generation === generationRef.current) stopStreaming();
    } finally {
      if (generation === generationRef.current) startingRef.current = false;
    }
  }, [stopStreaming]);

  const toggleStreaming = useCallback(() => {
    if (streamingRef.current) return stopStreaming();
    return startStreaming();
  }, [startStreaming, stopStreaming]);

  const toggleMute = useCallback(() => setIsMuted((prev) => !prev), []);

  // Callback estável: mudanças de estado não desmontam a transmissão.
  useEffect(() => () => stopStreaming(), [stopStreaming]);

  return {
    videoRef,
    isStreaming,
    isConnected,
    isMuted,
    processedFrame,
    metrics,
    events,
    yawnCount,
    toggleStreaming,
    toggleMute,
    stopStreaming,
  };
};
