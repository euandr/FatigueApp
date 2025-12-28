import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { AlertCircle, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center px-4 overflow-hidden relative">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-72 h-72 bg-red-500/20 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-72 h-72 bg-orange-500/20 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse delay-2000"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 max-w-md w-full">
        <Card className="bg-slate-800/50 border-slate-700 shadow-2xl">
          <div className="p-8">
            {/* Icon */}
            <div className="mb-8 flex justify-center">
              <div className="p-6 bg-gradient-to-br from-red-500/20 to-orange-500/20 rounded-full border border-red-500/30">
                <AlertCircle className="w-16 h-16 text-red-400 animate-bounce" />
              </div>
            </div>

            {/* 404 Text */}
            <h1 className="text-7xl font-black bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400 bg-clip-text text-transparent mb-4 text-center tracking-tighter">
              404
            </h1>

            {/* Main message */}
            <h2 className="text-2xl font-bold text-white mb-3 text-center">
              Página não encontrada
            </h2>

            {/* Alert */}
            <Alert className="mb-6 bg-red-950/30 border-red-800">
              <AlertCircle className="h-4 w-4 text-red-500" />
              <AlertTitle className="text-red-400">Rota inválida</AlertTitle>
              <AlertDescription className="text-red-300/80 font-mono text-sm mt-2 break-all">
                {location.pathname}
              </AlertDescription>
            </Alert>

            {/* Buttons */}
            <div className="flex flex-col gap-3">
              <Button
                variant="outline"
                className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
                onClick={() => window.history.back()}
              >
                <ArrowLeft className="w-4 h-4" />
                Voltar
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default NotFound;
