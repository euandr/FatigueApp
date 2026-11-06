# Como está o backend agora

> Resumo do estado atual de `backend/app`, para consulta rápida.
> O documento detalhado do estado anterior é `notas/como_estava_o_backend.md`.
> A documentação de banco (`DATABASE_atualizado.md`) descreve o **modelo-alvo**,
> que **não está implementado** neste backend no momento.

---

## 1. Visão geral

Hoje o backend é uma API **FastAPI** mínima, focada apenas em **detecção de
fadiga por WebSocket**. Ele não tem autenticação, não acessa banco, não cria
sessões e não grava eventos.

| Canal | Rota | Papel |
| --- | --- | --- |
| HTTP | `/health` | Liveness simples (`{"status": "ok"}`) |
| WebSocket | `/ws` | Recebe frames JPEG e devolve o resultado da detecção |

O app é criado por `create_app()` em `main.py`, que exporta `app = create_app()`.

---

## 2. Arquivos atuais

```
backend/
├── .env                  # CORS_ORIGINS e MAX_FRAME_BASE64
├── .env.example
├── requirements.txt
├── detector_fadiga/      # código-referência original (não usado pela API)
└── app/
    ├── main.py           # app FastAPI, /health e /ws
    ├── config.py         # Config (CORS + limite de frame)
    ├── monitoring.py     # Monitor por conexão (executor + detector)
    └── detector.py       # MediaPipe: EAR/MAR, piscadas, bocejo, fadiga
```

Não existem mais `auth.py`, `database.py`, `routes.py`, `schemas.py` nem a
pasta `tests/` que apareciam na versão anterior.

---

## 3. Inicialização

`create_app()`:

1. carrega `Config()` (lê `backend/.env`);
2. valida apenas `MAX_FRAME_BASE64 > 0`;
3. cria o FastAPI com título `API Detec. F.`;
4. registra CORS com as origens configuradas (`allow_methods=["GET"]`);
5. guarda a config em `app.state.config`.

Não há `lifespan`, checagem de esquema, reconciliação de sessões, `sweep`
nem conexão com o Supabase. **O app sobe mesmo sem banco.**

---

## 4. Conexão WebSocket (fluxo)

O navegador não envia token. A conexão é aberta direto:

1. Cliente conecta em `ws://host:8000/ws`.
2. Servidor aceita e responde imediatamente:
   ```json
   { "type": "ready" }
   ```
3. Cliente envia um frame por mensagem:
   ```json
   { "frame": "<jpeg em base64, sem o prefixo data:image/jpeg;base64,>" }
   ```
4. Servidor processa e responde com as métricas + o frame reprocessado.

Mensagens de controle:

- `{"type": "auth", ...}` → **ignorada** (compatibilidade com o handshake antigo);
- `{"type": "ping"}` → responde `{"type": "pong"}`;
- JSON inválido ou mensagem sem `frame` → `{"type": "error", "message": "..."}`.

O processamento é **sequencial por conexão**: recebe → valida → processa →
responde. Não há timeout de recebimento nem encerramento por inatividade: a
conexão fica aberta até o cliente fechar.

### Validações do frame

- precisa ser um `dict` com `frame` do tipo `str` não vazio;
- o base64 não pode passar de `MAX_FRAME_BASE64` (padrão **2.000.000** caracteres);
- base64 inválido ou imagem que não decodifica → `{"type": "error", ...}`.

### Encerramento

- `WebSocketDisconnect` → fecha silenciosamente;
- qualquer outra exceção → log + `close(code=1011)`;
- no `finally`, o `Monitor` da conexão é liberado (detector + executor).

---

## 5. O que o backend faz com os frames

Cada conexão cria seu próprio `Monitor` (`monitoring.py`), com:

- um **`ThreadPoolExecutor(max_workers=1)`** — o MediaPipe precisa ser criado e
  usado sempre na mesma thread;
- um `Detector` criado de forma preguiçosa (na primeira imagem válida).

O processamento roda fora do event loop:

```python
result = await asyncio.get_running_loop().run_in_executor(
    monitor.executor, monitor.process, frame
)
```

`Monitor.process`:

1. `base64.b64decode(..., validate=True)`;
2. `cv2.imdecode(...)` → `ValueError("Imagem inválida")` se falhar;
3. cria o `Detector` na primeira chamada;
4. chama `Detector.process_frame(frame)`.

`Detector` (MediaPipe FaceMesh, 1 rosto, refinamento de landmarks):

- redimensiona para largura **640** mantendo a proporção e converte BGR → RGB;
- calcula **EAR** (olhos) e **MAR** (boca);
- histórico de EAR em janela de **30 frames** para contar piscadas;
- `eyesClosed`: `ear < 0.2`;
- `fatigueAlert`: **30 frames consecutivos** com olhos fechados;
- `totalBlinks`: incrementa a cada **3 frames** com olhos fechados;
- `excessBlinks`: **5 ou mais** piscadas na janela de 30;
- `yawnDetected`: `mar > 0.6`;
- reencoda o frame em JPEG qualidade **70** e devolve em base64.

Formato da resposta:

```json
{
  "ear": 0.28,
  "mar": 0.21,
  "blinks": 1,
  "totalBlinks": 12,
  "yawnDetected": false,
  "excessBlinks": false,
  "fatigueAlert": false,
  "frame": "<jpeg base64>"
}
```

Observações:

- quando **não** há rosto, o resultado é o "neutro" (`ear: 0.3`, `mar: 0.2`,
  flags falsas) com o frame reencodado;
- quando há rosto, o detector **não devolve `eyesClosed`**; o frontend deriva
  esse valor de `ear < 0.2`;
- o frame devolvido é o mesmo recebido (redimensionado/recomprimido), **sem
  desenho de landmarks ou overlays**.

---

## 6. O que foi removido em relação ao estado anterior

- Autenticação: sem JWT, sem `Authorization`, sem consulta ao Supabase Auth.
- Banco: sem `httpx`, sem PostgREST, sem `check_schema`, sem `reconcile`.
- Sessões: sem `/api/sessions`, sem estado de sessão em memória, sem `sweep`,
  sem timeout de reconexão.
- Câmeras: sem CRUD `/api/devices`.
- Eventos: nada é persistido. Os alertas existem **somente no cliente**.

O login com Supabase Auth continua no **frontend**, mas o backend não valida
nem usa esse token.

---

## 7. Frontend e conexão

`frontend/src/services/api.js` monta as URLs:

- `VITE_API_URL` (padrão `http://<host>:8000`);
- `VITE_WS_URL` (padrão `ws(s)://.../ws`).

`useFatigueDetection.js`:

- captura a webcam em **640×480**;
- envia JPEG qualidade **0.7** a cada **100 ms** (~10 FPS);
- usa **um frame pendente por vez** (backpressure): só envia o próximo depois
  da resposta anterior;
- espera `{"type": "ready"}` para liberar o envio;
- atualiza métricas e o frame processado;
- dispara som/eventos locais para fadiga, bocejo e excesso de piscadas
  (lista mantida apenas em memória, com limite de 50 itens);
- reconecta com backoff de 1 s até 10 s em quedas comuns;
- **não** reconecta após `1000` (normal) ou `1008` (recusado);
- `1000`/`1008` encerram o streaming; não há mais reconexão de sessão nem
  revalidação periódica de token.

---

## 8. Configuração e execução

`backend/.env`:

| Variável | Padrão | Uso |
| --- | --- | --- |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Origens do CORS HTTP |
| `MAX_FRAME_BASE64` | `2000000` | Limite do frame em caracteres base64 |

Dependências (`requirements.txt`): `fastapi`, `fastapi-cli`,
`uvicorn[standard]`, `python-dotenv`, `mediapipe==0.10.14`, `numpy<2`,
`opencv-contrib-python==4.10.0.84`, `scipy`.

Executar (na pasta `backend`):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-max-size 2097152
```

Frontend (na pasta `frontend`):

```powershell
npm run dev -- --host 0.0.0.0
```

Saúde em `http://localhost:8000/health`; docs em `http://localhost:8000/docs`.

---

## 9. Limitações atuais

- **Sem autenticação**: qualquer cliente que alcance `/ws` pode enviar frames.
- **Sem persistência**: eventos, sessões e histórico não são gravados; fechar a
  aba perde tudo.
- **Sem controle de câmera**: não há cadastro, habilitação/desabilitação nem
  bloqueio de duas conexões na mesma câmera.
- **Um processo**: o estado (detector por conexão) vive na memória; não usar
  múltiplos workers.
- **CPU-bound**: cada conexão ativa consome uma thread de detecção; não há
  limite de conexões nem fila.
- **Sem RTSP**: o vídeo depende da aba do navegador (webcam).
- **Sem timeout de inatividade**: conexão ociosa permanece aberta.

Em resumo: o backend atual é só o **motor de detecção**. Autenticação, sessões,
câmeras e gravação de eventos descritos em `DATABASE_atualizado.md` ainda
precisam ser (re)implementados.
