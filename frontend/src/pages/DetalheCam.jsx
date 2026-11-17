import VideoFeed from "@/components/VideoFeed";
import { useNavigate } from "react-router-dom";
import Sidebar from "@/components/Sidebar";
import MetricsPanel from "@/components/MetricsPanel";
import ControlPanel from "@/components/ControlPanel";
import ConfigCamModal from "@/components/ConfigCamModal";
import { getCameraById } from "@/api";

import "./Monitoramento.css";
import CardResumoMonitoramento from "@/components/CardResumoMonitoramento";
import { useParams } from "react-router-dom";
import { useEffect, useState } from "react";
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

// import {useCamera} from "../hooks/useCamera";
import { useCameras } from "../contexts/CameraContext";

const DetalheCam = () => {
  const [modalAberto, setModalAberto] = useState(false);
  const navigate = useNavigate();
  const { id: cameraId } = useParams();
  const [camera, setCamera] = useState(null);
  const [erroCamera, setErroCamera] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [erroStream, setErroStream] = useState("");

  const {
    camerasAtivas,
    startCamera,
    stopCamera,
    deteccoes,
    alertas,
    sessoes,
  } = useCameras();
  const cameraUrl = camerasAtivas[cameraId];
  const isActive = Boolean(cameraUrl);
  const metricas = deteccoes[cameraId];

  const sessao = sessoes[cameraId];
  const [agora, setAgora] = useState(Date.now());

  useEffect(() => {
    setAgora(Date.now());

    if (!sessao || sessao.encerradaEm !== null) return;
    const intervalo = setInterval(() => {
      setAgora(Date.now());
    }, 1000);

    return () => clearInterval(intervalo);
  }, [sessao]);

  const inicioFormatado = sessao
    ? new Date(sessao.iniciadaEm).toLocaleTimeString("pt-BR")
    : "—";
  const segundos = sessao
    ? Math.max(
        0,
        Math.floor(((sessao.encerradaEm ?? agora) - sessao.iniciadaEm) / 1000),
      )
    : 0;
  const horas = String(Math.floor(segundos / 3600)).padStart(2, "0");
  const minutos = String(Math.floor((segundos % 3600) / 60)).padStart(2, "0");
  const segundosRestantes = String(segundos % 60).padStart(2, "0");

  const duracaoFormatada = sessao
    ? `${horas}h${minutos}min${segundosRestantes}s`
    : "—";

  const alertasAtivos = [];
  if (metricas?.prolongedEyeClosure) {
    alertasAtivos.push("Olhos fechados prolongadamente");
  }

  if (metricas?.yawnDetected) {
    alertasAtivos.push("Bocejo detectado");
  }

  if (metricas?.excessBlinks) {
    alertasAtivos.push("Piscadas excessivas");
  }

  function alternarCamera() {
    setIsConnected(false);
    setErroStream("");
    if (isActive) stopCamera(cameraId);
    else startCamera(cameraId, camera?.stream_url);
  }

  useEffect(() => {
    let ativo = true;
    setCamera(null);
    setErroCamera("");
    getCameraById(cameraId)
      .then((encontrada) => {
        if (
          !encontrada ||
          typeof encontrada !== "object" ||
          Array.isArray(encontrada)
        )
          throw new Error("Resposta inválida da API");
        if (ativo) {
          setCamera(encontrada);
          setErroCamera("");
        }
      })
      .catch((erro) => {
        console.error("Erro ao buscar câmera:", erro);
        if (ativo)
          setErroCamera(
            erro.response?.status === 404
              ? "Câmera não encontrada."
              : "Não foi possível carregar a câmera.",
          );
      });
    return () => {
      ativo = false;
    };
  }, [cameraId]);

  const iconEventoTable = (tipo) => {
    switch (tipo) {
      case "olhos_fechados":
        return <EyeClosed className="w-5 h-5 text-yellow-500" />;
      case "bocejo":
        return <FaceGrinning className="w-5 h-5 text-[#9d3ef4]" />;
      case "piscadas_excessivas":
        return <EyeOff className="w-5 h-5 text-blue-500" />;
      default:
        return <TriangleAlert className="w-5 h-5 text-gray-500" />;
    }
  };

  const eventosRecents = alertas.filter(
    (alerta) => String(alerta.cameraId) === String(cameraId),
  );

  return (
    <div className="detalhe-cam flex h-[calc(100dvh/1.25)] min-h-0 overflow-hidden bg-[#0e1925] text-left font-[Arial,Helvetica,sans-serif] text-[#f7f8fc]">
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
              <h1 className="text-[28px] font-bold text-white">
                {camera?.name || "Detalhes da câmera"}
              </h1>
              <span
                className={`flex items-center gap-2 ${isConnected ? "text-[#0cda69] bg-[#133632]" : "text-[#C62929] bg-[#4B1616]"} rounded-xl px-[16px] py-1 text-[12px]`}
              >
                <span
                  className={`w-[8px] h-[8px] rounded-full ${isConnected ? "bg-[#0cda69]" : "bg-[#C62929]"}`}
                ></span>
                {isConnected
                  ? "Conectada"
                  : erroStream
                    ? "Erro de conexão"
                    : isActive
                      ? "Conectando"
                      : "Desconectada"}
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
                <Link size={20} /> {camera?.protocolo || "Stream"}
              </span>
              <span> | </span>
              <span className="flex gap-[10px]">
                {" "}
                <Clock12 size={20} />{" "}
                {sessao && sessao.encerradaEm !== null
                  ? "Última sessão iniciada às"
                  : "Sessão iniciada às"}{" "}
                {inicioFormatado}
              </span>
              <span> | </span>
              <span className="flex gap-[10px]">
                {" "}
                <Clock4 size={20} /> Duração: {duracaoFormatada}
              </span>
            </div>
          </div>
          <button
            onClick={() => setModalAberto(true)}
            className="flex text-[#A4C5DD] gap-[10px] border border-solid rounded-lg border-[#51607A] p-[10px] text-[14px]  bg-[#162434] hover:scale-105 transition-transform duration-200"
          >
            <SquarePen size={20} /> Editar Câmera
          </button>
        </header>

        {/* Main Content */}
        <main className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6 pt-2 px-4 md:pt-2 md:px-6 lg:pt-2 lg:px-8">
          {erroCamera && (
            <p role="alert" className="col-span-full text-red-400">
              {erroCamera}
            </p>
          )}
          {/* Left Column - Video Feed */}
          <div className="lg:col-span-2 lg:flex lg:flex-col">
            <VideoFeed
              alertActive={isActive && alertasAtivos.length > 0}
              cameraUrl={cameraUrl}
              isStreaming={isActive}
              isConnected={isConnected}
              cameraError={erroStream}
              onCameraLoad={() => {
                setIsConnected(true);
                setErroStream("");
              }}
              onCameraError={() => {
                setIsConnected(false);
                setErroStream(
                  "Não foi possível carregar a câmera. Verifique a rede e o certificado HTTPS.",
                );
              }}
              deviceName={camera?.name}
            />
          </div>

          {/* Right Column - Metrics & Events */}
          <div className="flex flex-col gap-3">
            {/* Control Panel - Top on mobile */}
            <MetricsPanel
              isStreaming={isActive && Boolean(metricas)}
              ear={metricas?.ear ?? 0}
              mar={metricas?.mar ?? 0}
              blinks={metricas?.blinks ?? 0}
              yawns={null}
              eyesClosed={
                metricas?.eyesClosed ?? (metricas ? metricas.ear < 0.2 : false)
              }
              yawnDetected={metricas?.yawnDetected ?? false}
              excessBlinks={metricas?.excessBlinks ?? false}
            />

            <ControlPanel
              isStreaming={isActive}
              isConnected={isConnected}
              isMuted={false}
              onStartStop={alternarCamera}
              startDisabled={!camera?.stream_url}
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
                  <h2>{duracaoFormatada}</h2>
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
                    <td className="px-6 py-2">
                      {" "}
                      {new Date(evento.horario).toLocaleTimeString("pt-BR")}
                    </td>
                    <td className="py-2 flex items-center gap-2">
                      {iconEventoTable(evento.tipo)}
                      {evento.tipo === "olhos_fechados" && "Olhos fechados"}
                      {evento.tipo === "bocejo" && "Bocejo"}
                      {evento.tipo === "piscadas_excessivas" &&
                        "Piscadas excessivas"}
                    </td>
                    <td className="py-2">{evento.mensagem}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </main>
      </div>
      {modalAberto && (
        <ConfigCamModal
          mode="edit"
          cameraId={cameraId}
          onClose={(cameraAtualizada) => {
            if (cameraAtualizada) setCamera(cameraAtualizada);
            setModalAberto(false);
          }}
        />
      )}
    </div>
  );
};

export default DetalheCam;
