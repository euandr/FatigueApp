import { useState, useRef, useCallback, useEffect } from "react";
import alarmSound from "@/assets/alarm-clock.mp3";
import { createSession, endSession } from "@/lib/sessions";
import { saveEvent } from "@/lib/events";
import { supabase } from "@/lib/supabase";

export const useFatigueDetection = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [userId, setUserId] = useState(null);

  // Detectar servidor automaticamente
  // - Se em localhost: ws://localhost:8765
  // - Se em servidor remoto: ws://seu-servidor.com:8765

  const getServerURL = () => {
    // prioridade: variável de ambiente (ngrok / produção)
    if (import.meta.env.VITE_WS_URL) {
      return import.meta.env.VITE_WS_URL;
    }

    // fallback: comportamento antigo (localhost na porta 8765 com /ws)
    const host = window.location.hostname;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    return `${protocol}://${host}:8765/ws`;
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
  const reconnectTimeoutRef = useRef(null);
  const reconnectDelayRef = useRef(1000); // Começar com 1 segundo
  const sessionIdRef = useRef(null); // Ref para sessionId (sincronizado via useEffect)

  // Create alarm sound
  useEffect(() => {
    audioRef.current = new Audio(alarmSound);
    return () => {
      if (audioRef.current) {
        audioRef.current = null;
      }
    };
  }, []);

  // Get current user ID
  useEffect(() => {
    const getUser = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      if (user) {
        setUserId(user.id);
      }
    };
    getUser();
  }, []);

  // Sync isMuted with ref
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  // Sync sessionId with ref
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  const playAlarm = useCallback(() => {
    if (!isMutedRef.current && audioRef.current) {
      audioRef.current.currentTime = 0;
      audioRef.current.play().catch(() => {});
    }
  }, []);

  const addEvent = useCallback(
    (message, type, metadata = {}) => {
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

      // Salvar no banco APENAS eventos críticos (que disparam alerta)
      const isCriticalEvent =
        type === "fatigue_alert" || type === "yawn" || type === "excess_blinks";

      if (isCriticalEvent && sessionIdRef.current) {
        // Construir value com a métrica associada
        let metricValue = type; // Padrão: tipo técnico

        if (type === "yawn" && metadata.mar !== undefined) {
          metricValue = `MAR: ${metadata.mar.toFixed(3)}`;
        } else if (type === "fatigue_alert" && metadata.ear !== undefined) {
          metricValue = `EAR: ${metadata.ear.toFixed(3)}`;
        } else if (
          type === "excess_blinks" &&
          metadata.blinks_count !== undefined
        ) {
          metricValue = `Blinks: ${metadata.blinks_count}`;
        }

        console.log(`💾 Salvando: ${message} → ${metricValue}`);
        // event_type = descrição, value = métrica
        saveEvent(sessionIdRef.current, message, metricValue).catch((error) => {
          console.error("❌ Erro ao salvar evento no BD:", error);
        });
      }

      // Tocar alerta se for evento crítico
      if (isCriticalEvent) {
        playAlarm();
      }
    },
    [playAlarm],
  );

  const scheduleReconnect = useCallback((streaming) => {
    // Limpar timeout anterior
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    // Apenas reconectar se ainda está streaming
    if (!streaming) {
      reconnectDelayRef.current = 1000; // Reset delay
      return;
    }

    const delay = reconnectDelayRef.current;
    console.log(`Tentando reconectar em ${delay}ms...`);

    reconnectTimeoutRef.current = setTimeout(() => {
      connectWebSocket(streaming);
    }, delay);

    // Aumentar delay exponencialmente até máximo de 30 segundos
    reconnectDelayRef.current = Math.min(
      reconnectDelayRef.current * 1.5,
      30000,
    );
  }, []);

  const connectWebSocket = useCallback(
    (streaming = isStreaming) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Resetar delay ao conectar com sucesso
        reconnectDelayRef.current = 1000;
        return;
      }

      try {
        wsRef.current = new WebSocket(SERVER_URL);

        wsRef.current.onopen = () => {
          console.log("WebSocket connected");
          setIsConnected(true);
          // Resetar delay ao conectar com sucesso
          reconnectDelayRef.current = 1000;
        };

        wsRef.current.onclose = () => {
          console.log("WebSocket disconnected");
          setIsConnected(false);
          // Agendar reconexão se ainda está streaming
          scheduleReconnect(streaming);
        };

        wsRef.current.onerror = (error) => {
          console.error("WebSocket error:", error);
          setIsConnected(false);
          // Agendar reconexão se ainda está streaming
          scheduleReconnect(streaming);
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
                "Alerta de fadiga detectado - olhos fechados por muito tempo",
                "fatigue_alert",
                { ear: data.ear, duration: "contínuo" },
              );
            }

            if (data.yawnDetected) {
              setYawnCount((prev) => prev + 1);
              addEvent("Bocejo detectado", "yawn", { mar: data.mar });
            }

            if (data.excessBlinks) {
              addEvent(
                `Excesso de piscadas detectado (${data.blinks} em 30 frames)`,
                "excess_blinks",
                { blinks_count: data.blinks, threshold: 30 },
              );
            }
          } catch (e) {
            console.error("Error parsing WebSocket message:", e);
          }
        };
      } catch (error) {
        console.error("Error creating WebSocket:", error);
        setIsConnected(false);
        // Agendar reconexão se ainda está streaming
        scheduleReconnect(streaming);
      }
    },
    [addEvent, scheduleReconnect, isStreaming],
  );

  const startStreaming = useCallback(async () => {
    try {
      // Create a new session when starting detection
      if (!userId) {
        console.error("User ID not available");
        return;
      }

      const session = await createSession(userId);
      if (session) {
        setSessionId(session.id);
        console.log("Session created:", session.id);
      }

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

      connectWebSocket(true);

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
  }, [connectWebSocket, userId]);

  const stopStreaming = useCallback(async () => {
    // Cancelar tentativas de reconexão
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    reconnectDelayRef.current = 1000; // Reset delay

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

    // End session when stopping detection
    if (sessionId) {
      try {
        await endSession(sessionId);
        console.log("Session ended:", sessionId);
        setSessionId(null);
      } catch (error) {
        console.error("Error ending session:", error);
      }
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
  }, [sessionId]);

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
