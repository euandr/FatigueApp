import { useRef, useEffect } from "react";
import { Activity, Eye, AlertTriangle, Clock } from "lucide-react";

function EventsPanel({ events }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [events]);

  const getEventIcon = (type) => {
    switch (type) {
      case "eyes_closed":
        return <Eye className="w-4 h-4" />;
      case "yawn":
        return <AlertTriangle className="w-4 h-4" />;
      case "blink":
        return <Activity className="w-4 h-4" />;
      case "excess_blinks":
        return <Activity className="w-4 h-4" />;
      case "fatigue_alert":
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const getEventColor = (type) => {
    switch (type) {
      case "eyes_closed":
        return "text-red-400 bg-red-500/10 border-red-500/30";
      case "yawn":
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/30";
      case "blink":
        return "text-blue-400 bg-blue-500/10 border-blue-500/30";
      case "excess_blinks":
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/30";
      case "fatigue_alert":
        return "text-red-400 bg-red-500/10 border-red-500/30";
      default:
        return "text-slate-400 bg-slate-500/10 border-slate-500/30";
    }
  };

  const formatTime = (date) => {
    const dateObj = date instanceof Date ? date : new Date(date);
    return dateObj.toLocaleTimeString("pt-BR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 h-full flex flex-col">
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-slate-700/50 flex items-center justify-center">
          <Activity className="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <h2 className="text-lg font-bold text-slate-100">Eventos Recentes</h2>
          <p className="text-xs text-slate-400 font-mono">
            {events.length} eventos registrados
          </p>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-2 pr-2">
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-slate-500">
            <Clock className="w-8 h-8 mb-2 opacity-50" />
            <p className="text-sm font-mono">Nenhum evento detectado</p>
          </div>
        ) : (
          events.map((event, index) => (
            <div
              key={event.id}
              className={`p-3 rounded-lg border transition-all ${getEventColor(
                event.type
              )} ${index === 0 ? "animate-slide-in" : ""}`}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">{getEventIcon(event.type)}</div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono font-medium truncate">
                    {event.message}
                  </p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs font-mono opacity-70">
                      {event.camera}
                    </span>
                    <span className="text-xs opacity-50">•</span>
                    <span className="text-xs font-mono opacity-70">
                      {formatTime(event.timestamp)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default EventsPanel;
