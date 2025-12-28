import { useRef, useEffect, forwardRef } from "react";
import { Camera, CameraOff } from "lucide-react";

const VideoFeed = forwardRef(
  ({ isConnected, isStreaming, processedFrame, alertActive }, ref) => {
    const canvasRef = useRef(null);

    useEffect(() => {
      if (processedFrame && canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          const img = new Image();
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
          };
          img.src = `data:image/jpeg;base64,${processedFrame}`;
        }
      }
    }, [processedFrame]);

    return (
      <div
        className={`relative overflow-hidden rounded-lg border transition-all duration-300 ${
          alertActive
            ? "border-red-500 shadow-lg shadow-red-500/20"
            : "border-slate-700"
        }`}
      >
        {/* Video container */}
        <div className="relative aspect-video bg-slate-800/50">
          {/* Raw video (hidden, used for capture) */}
          <video
            ref={ref}
            autoPlay
            playsInline
            muted
            className={`absolute inset-0 w-full h-full object-cover ${
              processedFrame ? "opacity-0" : "opacity-100"
            }`}
          />

          {/* Processed frame from Python */}
          {processedFrame && (
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover"
            />
          )}

          {/* Overlay when not streaming */}
          {!isStreaming && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 backdrop-blur-sm">
              <CameraOff className="w-16 h-16 text-slate-500 mb-4" />
              <p className="text-slate-400 font-mono">Câmera desligada</p>
            </div>
          )}

          {/* Connection status overlay */}
          {isStreaming && !isConnected && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/60 backdrop-blur-sm">
              <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mb-4" />
              <p className="text-blue-400 font-mono">
                Conectando ao servidor...
              </p>
            </div>
          )}
        </div>

        {/* Camera info bar */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-slate-900/90 to-transparent p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Camera className="w-4 h-4 text-blue-400" />
              <span className="text-sm font-mono text-slate-300">CAM_01</span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? "bg-green-500" : "bg-red-500"
                }`}
              />
              <span className="text-xs font-mono text-slate-400">
                {isConnected ? "ONLINE" : "OFFLINE"}
              </span>
            </div>
          </div>
        </div>

        {/* Alert overlay */}
        {alertActive && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 px-6 py-2 bg-red-600/90 rounded-full border border-red-500">
            <span className="font-bold text-white tracking-wider">
              ⚠ ALERTA DE FADIGA
            </span>
          </div>
        )}
      </div>
    );
  }
);

VideoFeed.displayName = "VideoFeed";

export default VideoFeed;
