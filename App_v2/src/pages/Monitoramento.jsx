import { Camera, LogOut } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import VideoFeed from "@/components/VideoFeed";
import MetricsPanel from "@/components/MetricsPanel";
import EventsPanel from "@/components/EventsPanel";
import ControlPanel from "@/components/ControlPanel";
import { useFatigueDetection } from "@/hooks/useFatigueDetection";
import { useDevices } from "@/hooks/useDevices";
import { supabase } from "@/lib/supabase";

//  de onde peguei esse arquivo estava salvo como Index.jsx

const Monitoramento = () => {
  const navigate = useNavigate();
  const [userEmail, setUserEmail] = useState("");

  // ✅ PRIMEIRO: Get devices hook
  const { currentDevice, registerDeviceOnCamera } = useDevices();

  // ✅ SEGUNDO: Use FatigueDetection WITH currentDevice.id
  const {
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
  } = useFatigueDetection({ deviceId: currentDevice?.id });

  // ✅ Handler para iniciar/parar com device pronto
  const handleStartStop = async () => {
    // Se está parando, só para
    if (isStreaming) {
      toggleStreaming();
      return;
    }

    // Se está iniciando e não tem device, registra PRIMEIRO
    if (!currentDevice) {
      console.log("📱 Registrando device...");
      const result = await registerDeviceOnCamera();
      if (!result.success) {
        console.error("Falha ao registrar device:", result.error);
        return;
      }
      console.log("✅ Device registrado:", result.device?.id);

      // Inicia streaming COM o deviceId registrado
      toggleStreaming(result.device?.id);
    } else {
      // Se já tem device, usa o ID dele
      console.log("🎬 Iniciando streaming com device:", currentDevice?.id);
      toggleStreaming(currentDevice?.id);
    }
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    navigate("/");
  };

  const alertActive =
    metrics.fatigueAlert || metrics.yawnDetected || metrics.excessBlinks;

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex flex-col overflow-auto">
      {/* Header */}
      <header className="w-full bg-slate-800/80 backdrop-blur-sm border-b border-slate-700/50 px-4 md:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-end">
          <button
            onClick={handleLogout}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 hover:border-emerald-500/50 rounded-lg transition-all duration-200 group"
          >
            <LogOut className="w-5 h-5 text-emerald-400 group-hover:text-emerald-300 transition-colors" />
            <span className="text-sm font-semibold text-emerald-400 group-hover:text-emerald-300 transition-colors">
              Sair
            </span>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6 p-4 md:p-6 lg:p-8">
        {/* Left Column - Video Feed */}
        <div className="lg:col-span-2 space-y-4 md:space-y-6">
          <VideoFeed
            ref={videoRef}
            isConnected={isConnected}
            isStreaming={isStreaming}
            processedFrame={processedFrame}
            alertActive={alertActive}
            deviceName={currentDevice?.name || "CAM_01"}
          />

          {/* Control Panel - Below video on desktop */}
          <div className="hidden lg:block">
            <ControlPanel
              isStreaming={isStreaming}
              isConnected={isConnected}
              isMuted={isMuted}
              onStartStop={handleStartStop}
              onMuteToggle={toggleMute}
            />
          </div>
        </div>

        {/* Right Column - Metrics & Events */}
        <div className="space-y-4 md:space-y-6">
          {/* Control Panel - Top on mobile */}
          <div className="lg:hidden">
            <ControlPanel
              isStreaming={isStreaming}
              isConnected={isConnected}
              isMuted={isMuted}
              onStartStop={handleStartStop}
              onMuteToggle={toggleMute}
            />
          </div>

          <MetricsPanel
            isStreaming={isStreaming}
            ear={metrics.ear}
            mar={metrics.mar}
            blinks={metrics.blinks}
            yawns={yawnCount}
            eyesClosed={metrics.eyesClosed}
            yawnDetected={metrics.yawnDetected}
            excessBlinks={metrics.excessBlinks}
          />

          <div className="h-[400px]">
            <EventsPanel events={events} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-700/50">
        <div className="flex items-center justify-start py-6 px-4 md:px-6 lg:px-8 text-sm text-slate-500 font-mono">
          <p>Detecção de Fadiga v1.0</p>
        </div>
      </footer>
    </div>
  );
};

export default Monitoramento;
