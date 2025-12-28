import { useState, useRef, useCallback, useEffect } from "react";

export const useFatigueDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  // CONFIGURE YOUR WEBSOCKET SERVER URL HERE:
  // const SERVER_URL = 'ws://localhost:8765';
  const SERVER_URL = "ws://localhost:8765";
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

  // Create alarm sound
  useEffect(() => {
    audioRef.current = new Audio(
      "data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2eleWM3PlJxn97s0qxbSEdnm9Lf5+/s4dHMs6lqNyouYZzs8Ork39nV2dnQwrtxPhUlOnGh4/Xy6OHb2NnY09jN2LVcJxoYO3+t4/fy5+Ha19fX1dTV1MKyXiwjIUaItuHz8uXh29jY2NfV1NTTxLNhLiQkS4285PHx5ODa2NjY19bV1dTGtGMwJSVOj7/m8fHk4NrY2NjX19bW1ce1ZDElJlGSwufx8eTg29nZ2djX19bWyLdmMycnU5XD5/Hx5OHb2dnZ2NfX19bJuGgzKChVl8Xo8fHl4dvZ2dnY2NjX1sq5aTQpKVeZxujx8eXh29nZ2djY2NjWy7tqNSoqWZvH6fHx5eHb2dnZ2NjY2NfMvGw2KytbnMjp8fHl4dvZ2dnZ2NjY18y9bTcsLF2eyenly8eXi3NrZ2dnY2NjYzb5uOC0tX6DJ6vLx5uLc2tnZ2dnY2NjOv284Li5go8rq8vHm4tza2dnZ2dnZ2M/Abzkvb2GkyuvzwOLc2tra2dnZ2djQwXE5MDBjpsvr8vLm4t3a2tra2dnZ2NHCcjo="
    );
    return () => {
      if (audioRef.current) {
        audioRef.current = null;
      }
    };
  }, []);

  const playAlarm = useCallback(() => {
    if (!isMuted && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {});
    }
  }, [isMuted]);

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
        type === "yawn"
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
