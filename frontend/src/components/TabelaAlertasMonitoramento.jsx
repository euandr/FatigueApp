import { ChevronRight, TriangleAlert } from "lucide-react";

export default function TabelaAlertasMonitoramento({ alertas = [] }) {
  return (
    <section className="rounded-[10px] border border-[#203040] bg-[#0f1c2a] px-[14px] pb-4 pt-[15px]" aria-labelledby="monitor-alerts-title">
      <h2 className="mb-[10px] flex items-center gap-[10px] text-[14px] font-bold" id="monitor-alerts-title">Histórico de Alertas <span className="inline-flex h-[18px] min-w-5 items-center justify-center rounded bg-[#ff090f] px-[5px] text-white">{alertas.length}</span></h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[650px] border-collapse text-left text-[10px] [&_th]:border-b [&_th]:border-[#21303f] [&_th]:px-[7px] [&_th]:pb-[7px] [&_th]:pt-1 [&_th]:font-semibold [&_th]:text-[#7185a2] [&_td]:whitespace-nowrap [&_td]:p-[7px] [&_td]:font-semibold" aria-label="Alertas de exemplo das câmeras">
          <thead>
            <tr>
              <th className="w-[19%]" scope="col">Câmera</th>
              <th className="w-[33%]" scope="col">Evento</th>
              <th scope="col">Horário</th>
              <th scope="col">Duração</th>
              <th className="w-[1%]" scope="col">Ações</th>
            </tr>
          </thead>
          <tbody>
            {alertas.map((alerta) => (
              <tr key={alerta.id}>
                <td>
                  <div className="flex items-center gap-[11px]">
                    <span className="grid h-8 w-8 place-items-center rounded-full bg-[#352932] text-[#ff454b]"><TriangleAlert className="h-[23px] w-[23px] fill-current stroke-[#352932]" aria-hidden="true" /></span>
                    {alerta.camera}
                  </div>
                </td>
                <td>{alerta.evento}</td>
                <td><time>{alerta.horario}</time></td>
                <td>{alerta.duracao}</td>
                <td>
                  <button type="button" className="hover:scale-105 transition-transform duration-200 inline-flex items-center gap-2 rounded-[6px] border border-[#263747] bg-transparent py-1 pl-[14px] pr-[9px] text-[10px] font-semibold text-[#f7f8fc]" aria-label={`Ver detalhes do alerta de ${alerta.camera}`}>
                    Ver detalhes<ChevronRight className="h-5 w-5 text-[#536b85]" aria-hidden="true" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
