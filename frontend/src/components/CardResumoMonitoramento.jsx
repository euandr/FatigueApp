export default function CardResumoMonitoramento({
  title,
  value,
  icon,
  backgroundColor = "#152e5b",
  iconColor = "#2374ff",
  local = null,
}) {
  return (
    <article className={`${local=== "DC" ? 'border border-[#627392] py-[20px]' : ''} flex min-h-[69px] items-center gap-[9px] rounded-lg bg-[#192533] px-3 py-[10px] max-[900px]:gap-2 max-[900px]:p-[10px] max-[640px]:flex-col max-[640px]:items-start`}>
      <span
        className="grid h-12 w-12 shrink-0 place-items-center rounded-xl [&_svg]:h-[30px] [&_svg]:w-[30px] [&_svg]:stroke-[1.5] max-[900px]:h-10 max-[900px]:w-9"
        style={{ backgroundColor, color: iconColor }}
        aria-hidden="true"
      >
        {icon}
      </span>
      <div>
        <h2 className="mb-[3px] text-[13px] font-semibold text-[#92a4c0] max-[900px]:text-[11px] max-[640px]:min-h-[26px]">{title}</h2>
        <p className="text-[28px] font-bold leading-none max-[640px]:text-[25px]">{value}</p>
      </div>
    </article>
  );
}
