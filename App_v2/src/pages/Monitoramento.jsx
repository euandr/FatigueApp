import { Camera, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import VideoFeed from "@/components/VideoFeed";
import MetricsPanel from "@/components/MetricsPanel";
import EventsPanel from "@/components/EventsPanel";
import ControlPanel from "@/components/ControlPanel";
import { useFatigueDetection } from "@/hooks/useFatigueDetection";

//  de onde peguei esse arquivo estava salvo como Index.jsx

const Monitoramento = () => {
  const navigate = useNavigate();
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
  } = useFatigueDetection();

  const alertActive =
    metrics.fatigueAlert || metrics.yawnDetected || metrics.excessBlinks;

  return (
    <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 h-screen flex flex-col overflow-auto">
      {/* Header */}
      <header className="w-full bg-slate-800/80 backdrop-blur-sm border-b border-slate-700/50 px-4 md:px-6 lg:px-8 py-6">
        <div className="flex items-center">
          <button
            onClick={() => navigate(-1)}
            className="flex items-center gap-2 px-4 py-2.5 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 hover:border-emerald-500/50 rounded-lg transition-all duration-200 group"
          >
            <ArrowLeft className="w-5 h-5 text-emerald-400 group-hover:text-emerald-300 transition-colors" />
            <span className="text-sm font-semibold text-emerald-400 group-hover:text-emerald-300 transition-colors">
              Voltar
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
          />

          {/* Control Panel - Below video on desktop */}
          <div className="hidden lg:block">
            <ControlPanel
              isStreaming={isStreaming}
              isConnected={isConnected}
              isMuted={isMuted}
              onStartStop={toggleStreaming}
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
              onStartStop={toggleStreaming}
              onMuteToggle={toggleMute}
            />
          </div>

          <MetricsPanel
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
