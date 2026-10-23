import { Eye, CircleDot, AlertTriangle } from "lucide-react";

const MetricsPanel = ({
  isStreaming,
  ear,
  mar,
  blinks,
  yawns,
  eyesClosed,
  yawnDetected,
  excessBlinks,
}) => {
  const getEarStatus = () => {
    if (!isStreaming)
      return {
        color: "text-slate-400",
        bg: "bg-slate-700/30",
        label: "INATIVO",
      };
    if (ear < 0.2)
      return { color: "text-red-500", bg: "bg-red-500/10", label: "FECHADOS" };
    if (ear < 0.25)
      return {
        color: "text-yellow-500",
        bg: "bg-yellow-500/10",
        label: "SONOLENTO",
      };
    return { color: "text-green-500", bg: "bg-green-500/10", label: "NORMAL" };
  };

  const getMarStatus = () => {
    if (!isStreaming)
      return {
        color: "text-slate-400",
        bg: "bg-slate-700/30",
        label: "INATIVO",
      };
    if (mar > 0.6)
      return { color: "text-red-500", bg: "bg-red-500/10", label: "BOCEJO" };
    if (mar > 0.4)
      return {
        color: "text-yellow-500",
        bg: "bg-yellow-500/10",
        label: "ABRINDO",
      };
    return { color: "text-green-500", bg: "bg-green-500/10", label: "NORMAL" };
  };

  const earStatus = getEarStatus();
  const marStatus = getMarStatus();

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 space-y-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-slate-700/50 flex items-center justify-center">
          <Eye className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-100">
            Métricas em Tempo Real
          </h2>
          <p className="text-xs text-slate-400 font-mono">
            Análise facial contínua
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* EAR Metric */}
        <div
          className={`p-4 rounded-lg ${earStatus.bg} border border-slate-600/50 transition-all duration-300`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-slate-400">
              Olhos (EAR)
            </span>
            <span className={`text-xs font-mono font-bold ${earStatus.color}`}>
              {earStatus.label}
            </span>
          </div>
          <div className={`text-2xl font-bold font-mono ${earStatus.color}`}>
            {ear.toFixed(3)}
          </div>
          <div className="mt-2 h-1 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                eyesClosed ? "bg-red-500" : "bg-blue-500"
              }`}
              style={{ width: `${Math.min((ear / 0.4) * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* MAR Metric */}
        <div
          className={`p-4 rounded-lg ${marStatus.bg} border border-slate-600/50 transition-all duration-300`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-slate-400"> Boca (MAR)</span>
            <span className={`text-xs font-mono font-bold ${marStatus.color}`}>
              {marStatus.label}
            </span>
          </div>
          <div className={`text-2xl font-bold font-mono ${marStatus.color}`}>
            {mar.toFixed(3)}
          </div>
          <div className="mt-2 h-1 bg-slate-700 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                yawnDetected ? "bg-red-500" : "bg-blue-500"
              }`}
              style={{ width: `${Math.min((mar / 1.0) * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* Blinks Counter */}
        <div
          className={`p-4 rounded-lg ${
            excessBlinks ? "bg-yellow-500/10" : "bg-slate-700/30"
          } border border-slate-600/50 transition-all duration-300`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-slate-400">Piscadas</span>
            {isStreaming ? (
              <CircleDot
                className={`w-4 h-4 ${
                  excessBlinks ? "text-yellow-500" : "text-slate-400"
                }`}
              />
            ) : (
              <span className="text-xs font-mono text-slate-400">Inativo</span>
            )}
          </div>
          <div
            className={`text-2xl font-bold font-mono ${
              excessBlinks ? "text-yellow-500" : "text-slate-300"
            }`}
          >
            {blinks}
          </div>
          <p className="text-xs text-slate-500 mt-1">últimos 30 frames</p>
        </div>

        {/* Yawns Counter */}
        <div
          className={`p-4 rounded-lg ${
            yawns > 0 ? "bg-orange-500/10" : "bg-slate-700/30"
          } border border-slate-600/50 transition-all duration-300`}
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-mono text-slate-400">Bocejos</span>
            {isStreaming ? (
              <AlertTriangle
                className={`w-4 h-4 ${
                  yawns > 2 ? "text-red-500" : "text-orange-400"
                }`}
              />
            ) : (
              <span className="text-xs font-mono text-slate-400">Inativo</span>
            )}
          </div>
          <div
            className={`text-2xl font-bold font-mono ${
              yawns > 2 ? "text-red-500" : "text-orange-400"
            }`}
          >
            {yawns}
          </div>
          <p className="text-xs text-slate-500 mt-1">por minuto</p>
        </div>
      </div>

      {/* Alert Indicators */}
      <div className="space-y-2">
        <h3 className="text-xs font-mono text-slate-500 uppercase tracking-wider">
          Status de Alerta
        </h3>
        <div className="flex flex-wrap gap-2">
          {!isStreaming ? (
            <div className="px-3 py-1.5 rounded-full text-xs font-mono bg-slate-700 text-slate-400">
              Câmera Inativa
            </div>
          ) : (
            <>
              <div
                className={`px-3 py-1.5 rounded-full text-xs font-mono transition-all ${
                  eyesClosed
                    ? "bg-red-600 text-white"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                Olhos Fechados
              </div>
              <div
                className={`px-3 py-1.5 rounded-full text-xs font-mono transition-all ${
                  yawnDetected
                    ? "bg-red-600 text-white"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                Bocejo
              </div>
              <div
                className={`px-3 py-1.5 rounded-full text-xs font-mono transition-all ${
                  excessBlinks
                    ? "bg-yellow-600 text-white"
                    : "bg-slate-700 text-slate-400"
                }`}
              >
                Excesso Piscadas
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default MetricsPanel;
