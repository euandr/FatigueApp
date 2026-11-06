import { MoreVertical, VideoOff, Power, SquarePen, Trash } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function CardCamera({
  name = "CAM_01",
  status = "Desconectada",
  setModalAberto,
  setMode,
}) {
  const [modalSettings, setModalSettings] = useState(false);
  const navigate = useNavigate();
  return (
    <article
      onClick={() => navigate(`/DetalheCam/${name}`)}
      className=" hover:scale-105 transition-transform duration-200 relative min-w-0 rounded-[6px] border border-[#263747] p-[3px]"
      aria-label={`Câmera ${name}: ${status}`}
    >
      <div className="flex aspect-[2.28/1] flex-col items-center justify-center gap-[9px] rounded-[5px] bg-[#152231] text-[#657997]">
        <VideoOff
          className="h-[35px] w-[35px] fill-current stroke-[#152231] stroke-[1.5]"
          aria-hidden="true"
        />
        <p className="text-[12px] font-semibold">Sem transmissão</p>
      </div>
      <div className="flex items-center justify-between gap-[10px] px-[6px] pb-[7px] pt-[9px]">
        <div>
          <h3 className="mb-[5px] text-[12px] font-bold leading-[1.2]">
            {name}
          </h3>
          <p className="flex items-center gap-[7px] text-[10px] text-[#6b7f9c]">
            <span
              className="h-[10px] w-[10px] shrink-0 rounded-full bg-[#6b7f9c]"
              aria-hidden="true"
            />
            {status}
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
          <button className="text-[#ff0000] transition-transform duration-200 hover:scale-105">
            <Power size={18} /> Desconectar
          </button>
          <button 
            onClick={() => {
              setModalAberto(true)
              setMode("edit")
            }} className="transition-transform duration-200 hover:scale-105">
            <SquarePen size={18} /> Editar câmera
          </button>
          <button className="text-[#ff0000] transition-transform duration-200 hover:scale-105">
            <Trash size={18} /> Excluir câmera
          </button>
        </div>
      )}
    </article>
  );
}
