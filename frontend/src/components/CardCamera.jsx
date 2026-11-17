import { MoreVertical, VideoOff, Power, SquarePen, Trash } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteCamera } from "../api";

import { useCameras } from "../contexts/CameraContext";

export default function CardCamera({
  id,
  name,
  stream_url,
  status = "Desconectada",
  setModalAberto,
  setMode,
  setCameraId,
  setAtualizarCameras,
  atualizarCameras,
}) {
  const [modalSettings, setModalSettings] = useState(false);
  const navigate = useNavigate();
  const { camerasAtivas, startCamera, stopCamera, deteccoes } = useCameras();
  const cameraUrl = camerasAtivas[id];
  const metricas = deteccoes[id];
  const alertaAtivo = Boolean(
    cameraUrl &&
      (metricas?.prolongedEyeClosure ||
        metricas?.yawnDetected ||
        metricas?.excessBlinks)
  );

  return (
    <article
      onClick={() => navigate(`/DetalheCam/${id}`)}
      className={`hover:scale-105 transition-transform duration-200 relative min-w-0 rounded-[6px] border p-[3px] ${alertaAtivo ? "border-red-500 shadow-lg shadow-red-500/30" : "border-[#263747]"}`}
      aria-label={`Câmera ${name}: ${alertaAtivo ? "Alerta: sinal de fadiga" : cameraUrl ? "Transmissão iniciada" : status}`}
    >
      <div className="relative flex aspect-[2.28/1] items-center justify-center overflow-hidden rounded-[5px] bg-[#152231] text-[#657997]">
        {cameraUrl ? (
          <img
            src={cameraUrl}
            alt={`Transmissão da câmera ${name}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex flex-col items-center justify-center gap-[9px]">
            <VideoOff
              className="h-[35px] w-[35px] fill-current stroke-[#152231] stroke-[1.5]"
              aria-hidden="true"
            />
            <p className="text-[12px] font-semibold">Sem transmissão</p>
          </div>
        )}
        {alertaAtivo && (
          <span className="absolute top-2 left-2 rounded bg-red-600 px-2 py-1 text-xs font-bold text-white">
            ALERTA: SINAL DE FADIGA
          </span>
        )}
      </div>

      <div className="flex items-center justify-between gap-[10px] px-[6px] pb-[7px] pt-[9px]">
        <div>
          <h3 className="mb-[5px] text-[12px] font-bold leading-[1.2]">
            {name}
          </h3>
          <p className="flex items-center leading-none  gap-[7px] text-[10px] text-[#6b7f9c]">
            <span
              className={`h-[10px] w-[10px] shrink-0 rounded-full ${cameraUrl ? "bg-[#0cda69]" : "bg-[#6b7f9c]"}`}
              aria-hidden="true"
            />
            {cameraUrl ? "Transmissão iniciada" : status}
          </p>
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setModalSettings(!modalSettings);
          }}
          type="button"
          className=" hover:scale-[1.15] transition-transform duration-200 grid h-6 w-[23px] place-items-center self-end rounded-[6px] border border-[#263747] bg-transparent p-0 text-[#89a1c0]"
          aria-label={`Opções da câmera ${name}`}
        >
          <MoreVertical className="h-[17px] w-[17px]" aria-hidden="true" />
        </button>
      </div>

      {modalSettings && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute bottom-[10px] right-10 z-50 flex flex-col gap-[15px] rounded-[9px] border border-[#263242] bg-[#101B27] p-[15px] [&_button]:flex [&_button]:items-center [&_button]:gap-[10px] [&_button]:text-[12px]  "
        >
          <button
            type="button"
            disabled={!cameraUrl && !stream_url}
            onClick={() => {
              if (cameraUrl) stopCamera(id);
              else startCamera(id, stream_url);
              setModalSettings(false);
            }}
            className={`${cameraUrl ? "text-[#ff0000]" : "text-[#0bc905]"} transition-transform duration-200 hover:scale-105`}
          >
            <Power size={18} /> {cameraUrl ? "Desconectar" : "Conectar"}
          </button>
          <button
            onClick={() => {
              setCameraId(id);
              setModalAberto(true);
              setMode("edit");
              setModalSettings(!modalSettings);
            }}
            className="transition-transform duration-200 hover:scale-105"
          >
            <SquarePen size={18} /> Editar câmera
          </button>
          <button
            onClick={() => {
              deleteCamera(id);
              setAtualizarCameras(!atualizarCameras);
              setModalSettings(!modalSettings);
            }}
            className="text-[#ff0000] transition-transform duration-200 hover:scale-105"
          >
            <Trash size={18} /> Excluir câmera
          </button>
        </div>
      )}
    </article>
  );
}
