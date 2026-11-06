import { Video, ChartNoAxesColumn, Settings, LogOut } from "lucide-react";

const navButtonClasses =
  "flex items-center gap-[10px] whitespace-nowrap rounded-[6px] border-0 bg-transparent px-[10px] py-[11px] text-left text-[17px] font-semibold text-[#f7f8fc] aria-[current=page]:bg-[#152f68] max-[900px]:gap-[7px] max-[900px]:pl-2 max-[900px]:text-[14px] max-[640px]:p-2 max-[640px]:text-[12px]";

export default function Sidebar() {
  return (
    <aside
      className="relative flex h-full min-h-0 w-[230px] shrink-0 flex-col overflow-hidden border-r border-[#192735] bg-[#0b1521] max-[900px]:w-[180px] max-[640px]:h-auto max-[640px]:w-full max-[640px]:border-b max-[640px]:border-r-0 max-[640px]:border-b-[#203040] max-[640px]:p-[14px]"
      aria-label="Menu principal"
    >
      <div className="mx-[18px] mb-[25px] mt-[25px] whitespace-nowrap text-[27px] font-[750] tracking-[-.6px] max-[900px]:ml-[14px] max-[900px]:text-[24px] max-[640px]:m-0 max-[640px]:mb-[14px]">
        <span className="text-[#388fdf] ">Fatigue</span>App
      </div>
         <hr className="mb-[42px] border-0 border-t border-[#203040] max-[640px]:hidden " />
      
      <nav
        className="flex flex-col gap-[17px] px-[11px] [&_svg]:h-[23px] [&_svg]:w-[23px] [&_svg]:shrink-0 [&_svg]:stroke-[1.5] max-[640px]:flex-row max-[640px]:flex-wrap max-[640px]:gap-[5px] max-[640px]:p-0 max-[640px]:[&_svg]:h-[18px] max-[640px]:[&_svg]:w-[18px]"
        aria-label="Navegação principal"
      >
        <button type="button" className={navButtonClasses} aria-current="page">
          <Video aria-hidden="true" />
          Monitoramento
        </button>
        <button type="button" className={navButtonClasses}>
          <ChartNoAxesColumn aria-hidden="true" />
          Análises Gerais
        </button>
        <button type="button" className={navButtonClasses}>
          <Settings aria-hidden="true" />
          Configurações
        </button>
      </nav>

      <div className="mt-auto px-[11px] pb-[11px] max-[640px]:absolute max-[640px]:right-4 max-[640px]:top-4 max-[640px]:p-0">
        <hr className="mb-[16px] border-0 border-t border-[#203040] max-[640px]:hidden " />
        <button
          type="button"
          className="  hover:bg-red-500/10 transition-colors duration-300 flex w-full items-center gap-[10px] rounded-[12px] border-0 bg-transparent px-[17px] py-[11px] text-left text-[17px] font-semibold text-[#f7f8fc] [&_svg]:h-[23px] [&_svg]:w-[23px] [&_svg]:shrink-0 [&_svg]:stroke-[1.5] max-[640px]:p-0 max-[640px]:text-[13px]"
        >
          <LogOut aria-hidden="true" />
          Sair
        </button>
      </div>
    </aside>
  );
}
