import { Video, Wifi, TriangleAlert, Search, Plus } from "lucide-react";
import CardResumoMonitoramento from "../components/CardResumoMonitoramento.jsx";
import CardCamera from "../components/CardCamera.jsx";
import Sidebar from "../components/Sidebar.jsx";
import TabelaAlertasMonitoramento from "../components/TabelaAlertasMonitoramento.jsx";
import "./Monitoramento.css";

// Dados ilustrativos: nenhuma câmera está conectada nesta versão da tela.
const cameras = ["CAM_01", "CAM_02", "CAM_03", "CAM_04", "CAM_05", "CAM_06"];
const alertas = [
  {
    id: 1,
    camera: "CAM_01",
    evento: "Olhos fechados por tempo prolongado",
    horario: "10:24:18",
    duracao: "10 segundos",
  },
];

export default function Monitoramento() {
  return (
    <div className="monitoramento flex h-[calc(100dvh/1.25)] min-h-0 overflow-hidden bg-[#0e1925] text-left font-[Arial,Helvetica,sans-serif] text-[#f7f8fc] max-[640px]:flex-col [&_button:focus-visible]:outline [&_button:focus-visible]:outline-2 [&_button:focus-visible]:outline-offset-[3px] [&_button:focus-visible]:outline-[#80afff] [&_input:focus-visible]:outline [&_input:focus-visible]:outline-2 [&_input:focus-visible]:outline-offset-[3px] [&_input:focus-visible]:outline-[#80afff]">
      <Sidebar />
      <main className="min-h-0 min-w-0 flex-1 overflow-y-auto overscroll-y-contain px-5 pb-[18px] pt-[14px] max-[900px]:px-4 max-[900px]:py-[18px]">
        <header className="mb-4 flex items-center justify-between gap-4 max-[640px]:flex-wrap">
          <div>
            <h1 className="mb-[2px] text-[29px] font-bold leading-[1.15] tracking-[-.4px] max-[640px]:text-[26px]">
              Monitoramento
            </h1>
            <p className="text-[14px] leading-[1.6] text-[#91a2be] max-[640px]:text-[12px]">
              Visão geral das câmeras e alertas em tempo real
            </p>
          </div>
          <button
            type="button"
            className="inline-flex items-center justify-center gap-[5px] whitespace-nowrap rounded-lg border-0 bg-[#2366f5] px-[11px] py-[9px] text-[11px] font-bold text-white hover:scale-105 hover:brightness-125 transition-all duration-200"
          >
            <Plus
              className="h-[15px] w-[15px] stroke-[3] "
              aria-hidden="true"
            />
            Adicionar câmera
          </button>
        </header>

        <section
          className="mb-[18px] grid grid-cols-3 gap-[10px] max-[640px]:gap-[7px]"
          aria-label="Resumo do monitoramento"
        >
          <CardResumoMonitoramento
            title="Câmeras cadastradas"
            value={cameras.length}
            icon={<Video />}
          />
          <CardResumoMonitoramento
            title="Monitorando"
            value={0}
            icon={<Wifi />}
            backgroundColor="#133632"
            iconColor="#0cda69"
          />
          <CardResumoMonitoramento
            title="Com alerta"
            value={alertas.length}
            icon={<TriangleAlert fill="#ff4149" stroke="#332631" />}
            backgroundColor="#332631"
            iconColor="#ff4149"
          />
        </section>

        <section
          className="mb-[22px] flex flex-wrap items-center gap-[5px] max-[640px]:gap-[10px]"
          aria-label="Busca e filtros de câmeras"
        >
          <label className="flex h-8 w-[251px] items-center gap-[13px] rounded-[6px] border border-[#263747] px-[9px] max-[640px]:w-full">
            <Search
              className="h-[17px] w-[17px] shrink-0 text-[#a5bbda]"
              aria-hidden="true"
            />
            <input
              className="min-w-0 w-full border-0 bg-transparent text-[11px] text-white placeholder:text-[#8295b2]"
              type="search"
              placeholder="Buscar câmera..."
              aria-label="Buscar câmera"
            />
          </label>
          <div
            className="flex flex-wrap gap-1 [&_button]:rounded-[10px] [&_button]:border [&_button]:border-[#263747] [&_button]:bg-transparent [&_button]:px-[13px] [&_button]:py-[7px] [&_button]:text-[11px] [&_button]:font-semibold [&_button]:text-[#f7f8fc] [&_button[aria-pressed=true]]:border-[#2366f5] [&_button[aria-pressed=true]]:bg-[#2366f5]"
            role="group"
            aria-label="Status das câmeras"
          >
            <button
              className="hover:scale-105 transition-transform duration-200"
              type="button"
              aria-pressed="false"
            >
              Monitorando
            </button>
            <button
              className="hover:scale-105 transition-transform duration-200"
              type="button"
              aria-pressed="false"
            >
              Desconectadas
            </button>
            <button
              className="hover:scale-105 transition-transform duration-200"
              type="button"
              aria-pressed="false"
            >
              Com Alerta
            </button>
          </div>
        </section>

        <section
          className="mb-5 grid grid-cols-[repeat(auto-fill,minmax(min(245px,100%),1fr))] gap-[19px] min-[1600px]:grid-cols-[repeat(auto-fill,minmax(300px,1fr))]"
          aria-label="Câmeras cadastradas"
        >
          {cameras.map((name) => (
            <CardCamera key={name} name={name} />
          ))}
        </section>

        <TabelaAlertasMonitoramento alertas={alertas} />
      </main>
    </div>
  );
}
