import VideoFeed from "@/components/VideoFeed";
import { useNavigate } from "react-router-dom";
import MetricsPanel from "@/components/MetricsPanel";
import Sidebar from "@/components/Sidebar";
import ControlPanel from "@/components/ControlPanel";
import { useFatigueDetection } from "@/hooks/useFatigueDetection";
import { useDevices } from "@/hooks/useDevices";
import "./Monitoramento.css";
import CardResumoMonitoramento from "@/components/CardResumoMonitoramento";
import { useParams } from "react-router-dom";

import {
  ArrowLeft,
  Video,
  Link,
  Clock4,
  Clock12,
  SquarePen,
  ChartNoAxesColumn,
  TriangleAlert,
  FaceGrinning,
  EyeOff,
  EyeClosed,
  RotateCcwClock,
} from "lucide-react";

const DetalheCam = () => {
  const navigate = useNavigate();
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

  // const handleLogout = async () => {
  //   await stopStreaming();
  //   await supabase.auth.signOut();
  //   navigate("/");
  // };
  const alertActive =
    metrics.fatigueAlert || metrics.yawnDetected || metrics.excessBlinks;

  const iconEventoTable = (evento) => {
    switch (evento) {
      case "Olhos fechados":
        return <EyeOff className="w-5 h-5 text-red-500" />;
      case "Bocejo":
        return <FaceGrinning className="w-5 h-5 text-yellow-500" />;
      case "Piscadas excessivas":
        return <EyeClosed className="w-5 h-5 text-blue-500" />;
      default:
        return <TriangleAlert className="w-5 h-5 text-gray-500" />;
    }
  };

  const eventosRecents = [
    {
      id: 1,
      horario: "14:23:01",
      evento: "Olhos fechados",
      detalhes: "EAR: 0.18 | Duração: 3.2 s",
    },
    {
      id: 2,
      horario: "14:25:12",
      evento: "Bocejo",
      detalhes: "Duração: 2.1 s",
    },
    {
      id: 3,
      horario: "14:27:30",
      evento: "Piscadas excessivas",
      detalhes: "MAR: 0.72",
    },
  ];

  const { name } = useParams();

  return (
    <div className="detalhe-cam flex h-[calc(100dvh/1.25)] min-h-0 overflow-hidden bg-[#0e1925] text-left font-[Arial,Helvetica,sans-serif] text-[#f7f8fc]">
      {/* Header
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
      </header> */}
      <Sidebar />

      <div className="min-h-0 min-w-0 flex-1 overflow-auto">
        <header className="flex items-start place-content-between ml-[35px] mt-[30px] mr-[50px]">
          <div>
            <div className="flex items-center gap-4">
              <button
                type="button"
                aria-label="Voltar para Monitoramento"
                onClick={() => navigate("/Monitoramento")}
              >
                <ArrowLeft className="text-[#A4C5DD] hover:scale-[1.2] transition-transform duration-200" />{" "}
              </button>
              <h1 className="text-[28px] font-bold text-white">{name}</h1>
              <span className="flex items-center gap-2 text-[#C62929] bg-[#4B1616] rounded-xl px-[16px] py-1 text-[12px]">
                <span className="w-[8px] h-[8px] rounded-full bg-[#C62929]"></span>
                Desconectada
              </span>
            </div>
            <div className="mt-1 flex gap-3 text-[12px] font-bold text-[#A4C5DD] [&_span]:items-center">
              <span className="flex  gap-[10px]">
                {" "}
                <Video size={20} /> Camera IP
              </span>
              <span> | </span>
              <span className="flex  gap-[10px]">
                {" "}
                <Link size={20} /> RTSP
              </span>
              <span> | </span>
              <span className="flex gap-[10px]">
                {" "}
                <Clock12 size={20} /> Sessão iniciada às 14:23{" "}
              </span>
              <span> | </span>
              <span className="flex gap-[10px]">
                {" "}
                <Clock4 size={20} /> Duração: 1h23min{" "}
              </span>
            </div>
          </div>
          <button className="flex text-[#A4C5DD] gap-[10px] border border-solid rounded-lg border-[#51607A] p-[10px] text-[14px]  bg-[#162434]">
            <SquarePen size={20} /> Editar Câmera
          </button>
        </header>

        {/* Main Content */}
        <main className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6 pt-2 px-4 md:pt-2 md:px-6 lg:pt-2 lg:px-8">
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

            {/* Control Panel - Below video on desktop
          <div className="hidden lg:block">
            <ControlPanel
              isStreaming={isStreaming}
              isConnected={isConnected}
              isMuted={isMuted}
              onStartStop={handleStartStop}
              onMuteToggle={toggleMute}
            />
          </div> */}
          </div>

          {/* Right Column - Metrics & Events */}
          <div className="flex flex-col gap-4 md:gap-6">
            {/* Control Panel - Top on mobile */}

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

            <ControlPanel
              isStreaming={isStreaming}
              isConnected={isConnected}
              isMuted={isMuted}
              onStartStop={handleStartStop}
              onMuteToggle={toggleMute}
            />
          </div>

          <section className="col-span-full rounded-lg bg-[#12202E] p-[16px]">
            <header className="flex place-content-between    ">
              <div className="flex items-center gap-4">
                <ChartNoAxesColumn size={50} strokeWidth={4} color="#3B8ADA" />
                <div>
                  <h1 className="text-[20px] font-bold text-white">
                    Resumo da Sessão
                  </h1>
                  <p className="text-[14px] text-[#A4C5DD] font-bold">
                    Estatísticas acumuladas desde o início da sessão
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2 ">
                <Clock4 size={30} strokeWidth={1} />
                <div>
                  <h2>01h23min</h2>
                  <h2 className="text-[14px] text-[#A4C5DD] ">
                    Tempo de monitoramento
                  </h2>
                </div>
              </div>
            </header>
            <div className="mt-[16px] grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <CardResumoMonitoramento
                title="Total de alertas"
                value={4}
                icon={<TriangleAlert />}
                backgroundColor="#312735"
                iconColor="#ff4149"
                local="DC"
              />
              <CardResumoMonitoramento
                title="Bocejos Detectados"
                value={4}
                icon={<FaceGrinning />}
                backgroundColor="#272648"
                iconColor="#9D3EF4"
                local="DC"
              />
              <CardResumoMonitoramento
                title="Piscadas excessivas"
                value={4}
                icon={<EyeOff />}
                local="DC"
              />
              <CardResumoMonitoramento
                title="Olhos fechados prolongadamente"
                value={4}
                icon={<EyeClosed />}
                backgroundColor="#2B2E2C"
                iconColor="#FACD47"
                local="DC"
              />
            </div>
          </section>

          <section className="col-span-full rounded-lg bg-[#12202E] p-[16px] mb-[18px]">
            <div className="flex items-center gap-2">
              <RotateCcwClock size={30} strokeWidth={3} color="#3B8ADA" />
              <h1 className="text-[20px] font-bold text-white">
                Eventos Recentes
              </h1>
            </div>
            <table className="w-full mt-[12px] text-[#A4C5DD] text-left rounded-md overflow-hidden border-separate border-spacing-0">
              <thead>
                <tr className="bg-[#1C293B]">
                  <th className="px-6 py-2">Horário</th>
                  <th>Evento</th>
                  <th>Detalhes</th>
                </tr>
              </thead>
              <tbody>
                {eventosRecents.map((evento) => (
                  <tr
                    key={evento.id}
                    className="odd:bg-[#101C2B] even:bg-[#1C293B]"
                  >
                    <td className="px-6 py-2">{evento.horario}</td>
                    <td className="py-2 flex items-center gap-2">
                      {iconEventoTable(evento.evento)} {evento.evento}
                    </td>
                    <td className="py-2">{evento.detalhes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </main>
      </div>
    </div>
  );
};

export default DetalheCam;
