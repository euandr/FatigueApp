import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
} from "react";
import { WS_URL } from "@/api";
import { identificarNovosAlertas } from "@/services/alertas";

const CameraContext = createContext();

function CameraDetector({ cameraId, url_stream, onDeteccao, onAlertas }) {
  useEffect(() => {
    const imagem = new Image();
    let socket;
    let intervalo;
    let aguardandoResposta = false;

    let estadosAnteriores = {};

    const canvas = document.createElement("canvas");
    const contexto = canvas.getContext("2d");

    imagem.crossOrigin = "anonymous";
 
    imagem.onload = () => {
      if (socket) return;
      socket = new WebSocket(WS_URL);

      socket.onmessage = (evento) => {
        let dados;
        try {
          dados = JSON.parse(evento.data);
        } catch (erro) {
          console.error("Resposta inválida do detector:", cameraId, erro);
          socket.close();
          return;
        }

        if (dados.type === "ready") {
          canvas.width = imagem.naturalWidth;
          canvas.height = imagem.naturalHeight;

          intervalo = setInterval(() => {
            if (
              aguardandoResposta ||
              socket.readyState !== WebSocket.OPEN ||
              !contexto
            ) {
              return;
            }

            try {
              contexto.drawImage(imagem, 0, 0, canvas.width, canvas.height);
              const frame = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];

              aguardandoResposta = true;
              socket.send(JSON.stringify({ frame }));
            } catch (erro) {
              aguardandoResposta = false;
              console.error("Erro ao enviar frame:", cameraId, erro);
              clearInterval(intervalo);
            }
          }, 100);

          console.log("Monitoramento iniciado:", cameraId);
          return;
        }

        if (dados.type === "error") {
          aguardandoResposta = false;
          console.error("Erro do detector:", cameraId, dados.message);
          return;
        }

        aguardandoResposta = false;
        const { frame, ...metricas } = dados;
        onDeteccao(cameraId, metricas);
        console.log("Resultado:", cameraId, metricas);

        const novosAlertas = identificarNovosAlertas(estadosAnteriores, metricas);

        onAlertas(cameraId, novosAlertas);
        estadosAnteriores = metricas;
      };

      socket.onerror = () => {
        console.error("Erro no WebSocket:", cameraId);
      };

      socket.onclose = () => {
        clearInterval(intervalo);
        console.log("WebSocket encerrado:", cameraId);
      };
    };

    imagem.onerror = () => {
      console.error("Erro ao carregar a câmera:", cameraId);
    };

    imagem.src = url_stream;

    return () => {
      clearInterval(intervalo);
      imagem.onload = null;
      imagem.onerror = null;
      imagem.src = "";

      if (socket) {
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }

      console.log("Detector encerrado:", cameraId);
    };
  }, [cameraId, url_stream, onDeteccao, onAlertas]);

  return null;
}









export function CameraProvider({ children }) {
  const [camerasAtivas, setCamerasAtivas] = useState({});

  const [deteccoes, setDeteccoes] = useState({});

  const [alertas, setAlertas] = useState([]);

  const [sessoes, setSessoes] = useState({});

  const adicionarAlertas = useCallback((cameraId, novosAlertas) => {
    if (novosAlertas.length === 0) return;
    const eventos = novosAlertas.map((alerta) => ({
      id: crypto.randomUUID(),
      cameraId,
      tipo: alerta.tipo,
      mensagem: alerta.mensagem,
      horario: new Date().toISOString(),
    }));
    setAlertas((anteriores) => [...eventos, ...anteriores].slice(0, 50));
  }, []);

  const atualizarDeteccao = useCallback((cameraId, metricas) => {
    setDeteccoes((anteriores) => ({
      ...anteriores,
      [cameraId]: metricas,
    }));
  }, []);

  function startCamera(cameraId, url_stream) {
    if (!url_stream) return;
    
    if (camerasAtivas[cameraId]) return;

    const agora = Date.now();

    setSessoes((anteriores) => ({
      ...anteriores,
      [cameraId]:{iniciadaEm:agora, encerradaEm:null}
    }))

    setCamerasAtivas((anteriores) => ({
      ...anteriores,
      [cameraId]: url_stream,
    }));
  }

  function stopCamera(cameraId) {

    const agora = Date.now();

    setSessoes((anteriores) => {
      const sessao =anteriores[cameraId];
      if (!sessao || sessao.encerradaEm !==null) return anteriores;

      return {
        ...anteriores,
        [cameraId]: { ...sessao, encerradaEm: agora },
      }
    })
    setCamerasAtivas((anteriores) => {
      const atualizadas = { ...anteriores };
      delete atualizadas[cameraId];
      return atualizadas;
    });
    setDeteccoes((anteriores) => {
      const atualizadas = { ...anteriores };
      delete atualizadas[cameraId];
      return atualizadas;
    });
  }

  return (
    <CameraContext.Provider
      value={{ camerasAtivas, startCamera, stopCamera, deteccoes, alertas, sessoes }}
    >
      {Object.entries(camerasAtivas).map(([cameraId, url_stream]) => (
        <CameraDetector
          key={cameraId}
          cameraId={cameraId}
          url_stream={url_stream}
          onDeteccao={atualizarDeteccao}
          onAlertas={adicionarAlertas}
        />
      ))}

      {children}
    </CameraContext.Provider>
  );
}

export function useCameras() {
  return useContext(CameraContext);
}
