import { useState, useRef, useCallback, useEffect } from "react";
import alarmSound from "@/assets/alarm-clock.mp3";

export const useFatigueDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);

  // Detectar servidor automaticamente
  // - Se em localhost: ws://localhost:8765
  // - Se em servidor remoto: ws://seu-servidor.com:8765
  const getServerURL = () => {
    const host = window.location.hostname;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${host}:8765`;
  };

  const SERVER_URL = getServerURL();
  const [processedFrame, setProcessedFrame] = useState(null);
  const [events, setEvents] = useState([]);
  const [yawnCount, setYawnCount] = useState(0);

  const [metrics, setMetrics] = useState({
    ear: 0.3,
    mar: 0.2,
    blinks: 0,
    totalBlinks: 0,
    eyesClosed: false,
    yawnDetected: false,
    excessBlinks: false,
    fatigueAlert: false,
  });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const audioRef = useRef(null);
  const lastAlertRef = useRef({});
  const yawnResetIntervalRef = useRef(null);
  const isMutedRef = useRef(false);

  // Create alarm sound
  useEffect(() => {
    audioRef.current = new Audio(alarmSound);
    return () => {
      if (audioRef.current) {
        audioRef.current = null;
      }
    };
  }, []);

  // Sync isMuted with ref
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  const playAlarm = useCallback(() => {
    if (!isMutedRef.current && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {});
    }
  }, []);

  const addEvent = useCallback(
    (type, message) => {
      const now = Date.now();
      const cooldown = 2000; // 2 seconds cooldown between same event types

      if (
        lastAlertRef.current[type] &&
        now - lastAlertRef.current[type] < cooldown
      ) {
        return;
      }

      lastAlertRef.current[type] = now;

      const event = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type,
        camera: "camera1",
        timestamp: new Date(),
        message,
      };

      setEvents((prev) => [event, ...prev].slice(0, 50));

      if (
        type === "fatigue_alert" ||
        type === "eyes_closed" ||
        type === "yawn" ||
        type === "excess_blinks"
      ) {
        playAlarm();
      }
    },
    [playAlarm]
  );

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      wsRef.current = new WebSocket(SERVER_URL);

      wsRef.current.onopen = () => {
        console.log("WebSocket connected");
        setIsConnected(true);
      };

      wsRef.current.onclose = () => {
        console.log("WebSocket disconnected");
        setIsConnected(false);
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsConnected(false);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          setMetrics(data);

          if (data.frame) {
            setProcessedFrame(data.frame);
          }

          // Generate events based on detection
          if (data.fatigueAlert) {
            addEvent(
              "fatigue_alert",
              "Alerta de fadiga detectado - olhos fechados por muito tempo"
            );
          }

          if (data.yawnDetected) {
            setYawnCount((prev) => prev + 1);
            addEvent("yawn", "Bocejo detectado");
          }

          if (data.excessBlinks) {
            addEvent(
              "excess_blinks",
              `Excesso de piscadas detectado (${data.blinks} em 30 frames)`
            );
          }

          if (data.eyesClosed && !data.fatigueAlert) {
            addEvent(
              "eyes_closed",
              `Olhos fechados detectados (EAR: ${data.ear.toFixed(3)})`
            );
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      };
    } catch (error) {
      console.error("Error creating WebSocket:", error);
    }
  }, [addEvent]);

  const startStreaming = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Create canvas for capturing frames
      canvasRef.current = document.createElement("canvas");
      canvasRef.current.width = 640;
      canvasRef.current.height = 480;

      connectWebSocket();

      // Start sending frames
      intervalRef.current = window.setInterval(() => {
        if (
          videoRef.current &&
          canvasRef.current &&
          wsRef.current?.readyState === WebSocket.OPEN
        ) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, 640, 480);
            const frameData = canvasRef.current.toDataURL("image/jpeg", 0.7);
            const base64Data = frameData.split(",")[1];
            wsRef.current.send(JSON.stringify({ frame: base64Data }));
          }
        }
      }, 100); // ~10 FPS

      // Reset yawn count every minute (60000 ms)
      yawnResetIntervalRef.current = window.setInterval(() => {
        setYawnCount(0);
      }, 60000);

      setIsStreaming(true);
    } catch (error) {
      console.error("Error starting stream:", error);
    }
  }, [connectWebSocket]);

  const stopStreaming = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (yawnResetIntervalRef.current) {
      clearInterval(yawnResetIntervalRef.current);
      yawnResetIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    // Reset metrics and yawn count when stopping stream
    setMetrics({
      ear: 0,
      mar: 0,
      blinks: 0,
      totalBlinks: 0,
      eyesClosed: false,
      yawnDetected: false,
      excessBlinks: false,
      fatigueAlert: false,
    });
    setYawnCount(0);

    setIsStreaming(false);
    setIsConnected(false);
    setProcessedFrame(null);
  }, []);

  const toggleStreaming = useCallback(() => {
    if (isStreaming) {
      stopStreaming();
    } else {
      startStreaming();
    }
  }, [isStreaming, startStreaming, stopStreaming]);

  const toggleMute = useCallback(() => {
    setIsMuted((prev) => !prev);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopStreaming();
    };
  }, [stopStreaming]);

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
  };
};
