import {
  Play,
  Square,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
  Settings,
} from "lucide-react";
import { Button } from "./ui/button";

const ControlPanel = ({
  isStreaming,
  isConnected,
  isMuted,
  onStartStop,
  onMuteToggle,
}) => {
  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 space-y-4">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-slate-700/50 flex items-center justify-center">
          <Settings className="w-5 h-5 text-slate-400" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-100">Controles</h2>
          <p className="text-xs text-slate-400 font-mono">Gerenciar detecção</p>
        </div>
        <div
          className={`ml-auto flex items-center justify-center w-10 h-10 rounded-lg ${
            isConnected ? "bg-green-500/20" : "bg-slate-700/50"
          }`}
        >
          {isConnected ? (
            <Wifi className="w-5 h-5 text-green-500" />
          ) : (
            <WifiOff className="w-5 h-5 text-slate-500" />
          )}
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-3">
        <Button
          onClick={onStartStop}
          className={`flex-1 gap-2 font-mono transition-all ${
            isStreaming
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
        >
          {isStreaming ? (
            <>
              <Square className="w-4 h-4" />
              Parar
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Iniciar
            </>
          )}
        </Button>

        <Button
          onClick={onMuteToggle}
          variant="outline"
          className={`w-12 border-slate-600 ${
            isMuted ? "text-slate-500" : "text-blue-400 border-blue-500/50"
          }`}
        >
          {isMuted ? (
            <VolumeX className="w-5 h-5" />
          ) : (
            <Volume2 className="w-5 h-5" />
          )}
        </Button>
      </div>

      {/* Status Info */}
      <div className="pt-4 border-t border-slate-700/50">
        <div className="grid grid-cols-2 gap-4 text-center">
          <div>
            <p className="text-xs font-mono text-slate-500 uppercase">Status</p>
            <p
              className={`text-sm font-mono font-bold ${
                isStreaming ? "text-green-500" : "text-slate-500"
              }`}
            >
              {isStreaming ? "Ativo" : "Inativo"}
            </p>
          </div>
          <div>
            <p className="text-xs font-mono text-slate-500 uppercase">
              Conexão
            </p>
            <p
              className={`text-sm font-mono font-bold ${
                isConnected ? "text-green-500" : "text-red-500"
              }`}
            >
              {isConnected ? "Conectado" : "Desconectado"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
