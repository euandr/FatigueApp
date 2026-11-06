import { Link, Link2, Plus,Info  } from "lucide-react";

export default function AddCamModal({ onClose, mode }) {
  return (
    <div
      id="blur-fundo-modal"
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50"
    >
      <div className="w-[480px] max-w-[calc(100%-2rem)] bg-[#0B1522] rounded-xl border-[#243243] border-[1.8px]">
        <div className="p-6">
          <h1 className="text-2xl font-bold">{mode=="add"?"Adicionar Câmera":"Editar Câmera"}</h1>
          <p className=" text-[#525B67]">{mode=="add"?"Cadastre uma nova fonte de vídeo para monitoramento":"Edite a fonte de vídeo para monitoramento"}</p>
        </div>
        <hr className="border-[#243243]" />
        <div className="px-6 my-4 flex flex-col">
          <label htmlFor="camera-name" className="mb-1 ">
            Identificação da câmera
          </label>

          <div className="rounded-sm bg-[#1E2635] border-[1.8px] border-solid border-[#243243] px-2 py-1 focus-within:outline focus-within:outline-2 focus-within:outline-white">
            <input
              id="camera-name"
              type="text"
              placeholder="Ex: CAM_01"
              className="w-full border-0 bg-transparent text-[12px] !font-normal !outline-none !shadow-none focus:!outline-none focus:!ring-0 placeholder:text-[12px]"
            />
          </div>
          <label htmlFor="connection-type" className="mt-2 mb-1 ">
            Protocolo de conexão
          </label>
          <div className="relative">
            <Link
              size={18}
              className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none"
            />

            <select
              id="connection-type"
              defaultValue="http"
              className="w-full rounded-sm  bg-[#1E2635] border-[#243243] border-[1.8px] pr-32 pl-10 py-2 text-[12px] "
            >
              <option value="http">HTTP/HTTPS</option>
            </select>
          </div>

          <label htmlFor="stream-url" className="mt-2 mb-1  ">
            URL de transmissão
          </label>
          <div className="flex items-center gap-4 rounded-sm bg-[#1E2635] border-[#243243] border-[1.8px] px-2 py-1 focus-within:outline focus-within:outline-2 focus-within:outline-white">
            <Link2 size={18} />
            <input
              className="min-w-0 flex-1 border-0 bg-transparent !outline-none !shadow-none focus:!outline-none focus:!ring-0  placeholder:text-[12px] !font-normal"
              id="stream-url"
              type="url"
              placeholder="Ex: http://198.168.1.100:8080"
            ></input>
          </div>
        </div>
        <hr className="border-[#243243] mt-8" />
        <div className="flex justify-end gap-4 px-6 p my-4">
          <button onClick={onClose} className=" bg-[#1E2635] border-[#243243] border-[1.8px] rounded-sm px-4 py-1 hover:scale-105 transition-transform duration-200">
            Cancelar
          </button>
          <button onClick={onClose} className="flex bg-[#3259B9] rounded-sm px-4 py-1 items-center gap-2 hover:scale-105 transition-transform duration-200">
            {mode=="add"?(<><Plus size={16}/> "Adicionar câmera"</>):"Salvar"}
          </button>
        </div>
      </div>
    </div>
  );
}
