# Migração das comunicações para FastAPI

## O que mudou

- Nova aplicação em `backend/app`. A pasta `backend/detector_fadiga` não foi alterada.
- Operações do banco passam pela API. Supabase Auth continua responsável pelo login, cadastro, recuperação de senha e renovação do token no React.
- O backend valida o JWT consultando `GET /auth/v1/user` do próprio Supabase (assinatura, expiração e identidade são verificadas pelo serviço). A identidade enviada em campos do corpo não é aceita como autorização.
- WebSocket FastAPI em `/ws`, substituindo o servidor antigo na execução da aplicação migrada.
- Câmeras compartilhadas por todos os usuários autenticados, conforme escolha durante a implementação. Não existe `devices.user_id` nesta implementação. Qualquer usuário autenticado pode cadastrar, listar, habilitar e desabilitar câmeras. Sessões e eventos são consultáveis apenas pelo usuário que iniciou a sessão; o encerramento direto também exige esse usuário. Desabilitar uma câmera compartilhada encerra seu monitoramento, inclusive se iniciado por outro usuário.
- Eventos são produzidos e persistidos pelo backend, não por mensagens arbitrárias do navegador. A persistência ocorre no início de cada condição e é rearmada quando ela deixa de ocorrer. Isso evita uma linha a cada frame; não altera os cálculos nem os alertas transmitidos.
- Aparência, páginas, componentes e estilos preservados. O hook de comunicação foi reorganizado para autenticar o socket, evitar reconexão após parar e limpar os recursos da câmera.

## Arquivos criados

| Arquivo | Responsabilidade |
|---|---|
| `backend/app/__init__.py` | Pacote da aplicação |
| `backend/app/main.py` | FastAPI, CORS, ciclo de vida e WebSocket |
| `backend/app/config.py` | Configuração por ambiente |
| `backend/app/auth.py` | Dependência de autenticação REST |
| `backend/app/database.py` | HTTP para Supabase Auth/PostgREST e checagem de colunas |
| `backend/app/schemas.py` | Validação dos corpos das requisições |
| `backend/app/routes.py` | Endpoints REST |
| `backend/app/monitoring.py` | Sessões, desconexão, estado por câmera e persistência de eventos |
| `backend/app/detector.py` | Algoritmo original adaptado para estado por instância |
| `backend/requirements.txt` | Dependências específicas da aplicação migrada |
| `backend/.env.example` | Modelo de configuração privada |
| `backend/tests/test_api.py` | Autorização, REST, WebSocket, reconexão e ciclo de vida |
| `backend/tests/test_database.py` | Validação remota do JWT e consultas de esquema com transporte simulado |
| `backend/tests/test_detector.py` | Comparação com o detector original e processamento MediaPipe real sem rosto |
| `frontend/src/services/api.js` | Cliente HTTP autenticado e URLs da API/WebSocket |
| `frontend/.env.example` | Modelo de configuração pública |

Arquivos modificados: `.gitignore`, `frontend/src/services/queries/devices.js`, `frontend/src/services/sessions.js`, `frontend/src/services/events.js`, `frontend/src/hooks/useDevices.js`, `frontend/src/hooks/useFatigueDetection.js` e `frontend/src/pages/Monitoramento.jsx` (encerrar a sessão antes do logout). Este relatório também foi criado em `notas/MIGRACAO_FASTAPI.md`.

Os antigos helpers de escrita/exclusão de eventos, sem uso nas páginas, não são disponibilizados na API. Apenas o detector escreve eventos. As operações de leitura existentes continuam disponíveis pelo serviço de eventos.

## Dependências e ambiente

Foi criado um ambiente isolado `backend/.venv`; não se reutiliza nem altera o ambiente do detector de referência.

Dependências diretas declaradas e instaladas por `backend/requirements.txt`:

- FastAPI e Uvicorn com suporte WebSocket;
- HTTPX (Auth e PostgREST, sem SDK Supabase Python adicional);
- python-dotenv;
- MediaPipe `0.10.14`, OpenCV contrib `4.10.0.84`, SciPy e NumPy `1.x`, para manter a API `mp.solutions.face_mesh` e compatibilidade binária.

Não foram acrescentadas dependências no frontend. O pacote `websockets` pode ser instalado como dependência de transporte do Uvicorn; não existe servidor independente com `websockets.serve` na aplicação nova.

Versões diretas efetivamente instaladas nesta execução: FastAPI `0.156.2`, Uvicorn `0.54.0`, HTTPX `0.28.1`, python-dotenv `1.2.4`, MediaPipe `0.10.14`, NumPy `1.26.4`, OpenCV contrib `4.10.0.84` e SciPy `1.17.1`.

## Configurar e executar (PowerShell)

Na raiz do projeto, com Python 3.11:

```powershell
python -m venv backend/.venv
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
```

Preencha `backend/.env` com a URL e a chave **service_role** privada do projeto Supabase. A API não inicia sem essas configurações. Não coloque essa chave em nenhuma variável `VITE_*`, URL ou arquivo versionado.

Em `frontend/.env.local`, preserve as variáveis públicas existentes e configure:

```dotenv
VITE_API_URL=http://localhost:8000
# Remova uma VITE_WS_URL antiga ou atualize para:
VITE_WS_URL=ws://localhost:8000/ws
```

Não sobrescreva o `.env.local` existente com o exemplo sem preservar sua configuração de Auth.

Terminal do backend:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --ws-max-size 2097152
```

Alternativa para desenvolvimento (CLI `fastapi-cli` instalado no ambiente e incluído no `requirements.txt`): dentro de `backend`, execute `.\.venv\Scripts\fastapi.exe dev --entrypoint app.main:app`. O modo dev usa recarga automática; não use durante monitoramentos reais. É necessário configurar `backend/.env` antes de iniciar.

Terminal do frontend:

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Abra `http://localhost:5173`. A API expõe saúde em `http://localhost:8000/health` e documentação em `http://localhost:8000/docs`.

Em outro dispositivo, use o IP/nome do servidor em `VITE_API_URL` e inclua a origem exata do frontend em `CORS_ORIGINS` no backend. Para webcam fora de localhost, o navegador exige contexto seguro HTTPS; nesse caso use HTTPS na API e WSS no socket. Não execute em paralelo o servidor antigo para testar a nova integração.

## Banco: compatibilidade e segurança

Não foi executado SQL, não foram criadas colunas/tabelas/políticas e não houve migração automática. Não havia acesso configurado ao Supabase real na sessão de implementação; a validação real depende das credenciais no `.env`.

Na inicialização são feitas consultas **somente de leitura** das colunas documentadas. Se `events.metadata` ainda não existir, a inicialização com `EVENT_SCHEMA=numeric` falhará. Para o esquema antigo (`value` text e sem `metadata`), configure explicitamente `EVENT_SCHEMA=legacy`, sem alterar o banco. Nesse modo são salvos `MAR: 0.700`, `EAR: 0.100` e `Blinks: 5`. No modo `numeric`, são salvos número e `metadata.metric`. A checagem de colunas não comprova o tipo de `value`, constraints, defaults ou políticas: confira isso no painel Supabase antes do teste integrado. Não há fallback silencioso quando uma gravação falha.

**Atenção:** depois da checagem, a API preenche `ended_at` das sessões abertas que sobraram de execuções anteriores. Isso é reconciliação de dados solicitada, não mudança de estrutura. Use primeiro um projeto de teste ou encerre os monitoramentos antigos antes de iniciar esta API.

A chave de backend contorna RLS; por isso todos os endpoints de dados exigem autenticação e consultas a sessões/eventos aplicam `sessions.user_id` validado pelo Auth. Campos de câmera retornados são limitados a `id`, `name`, `created_at` e `is_active`, evitando expor configurações/credenciais de fontes futuras.

Revisar RLS e permissões no Supabase é obrigatório antes de produção: remover chamadas do frontend **não impede** um cliente externo de acessar PostgREST com a chave pública se as políticas atuais permitirem. Não se presume que a cadeia antiga `devices.user_id` exista. Para acesso exclusivo pela API, restrinja o acesso direto às três tabelas para os papéis públicos/autenticados, mantendo Auth. Nenhuma política foi alterada automaticamente.

## Endpoints

Todos abaixo exigem `Authorization: Bearer <access_token>` do Supabase:

| Método | Rota | Uso |
|---|---|---|
| GET | `/api/devices?active=true` | Câmeras habilitadas (omita o filtro para todas) |
| POST | `/api/devices` | Criar/obter câmera habilitada por nome, corpo `{ "name": "CAM_01" }` |
| GET | `/api/devices/{id}` | Consultar câmera |
| PATCH | `/api/devices/{id}` | Atualizar `name` e/ou `is_active` |
| POST | `/api/sessions` | Iniciar com `{ "device_id": "uuid" }` |
| GET | `/api/sessions?active=true` | Sessões do usuário, opcionalmente abertas |
| POST | `/api/sessions/{id}/end` | Encerrar sua sessão (idempotente) |
| GET | `/api/sessions/{id}/events` | Eventos da sua sessão |

### WebSocket

1. Conectar a `/ws`.
2. Enviar em até 10 segundos `{ "type": "auth", "token": "JWT", "session_id": "uuid" }`. O token não vai na URL.
3. Aguardar `{ "type": "ready", "session_id": "uuid" }`.
4. Enviar `{ "frame": "JPEG em base64" }`. A resposta mantém os campos originais: `ear`, `mar`, `blinks`, `totalBlinks`, `yawnDetected`, `excessBlinks`, `fatigueAlert`, `frame` (e `eyesClosed` quando já retornado no protótipo). Mensagens de controle usam `type` e não substituem métricas no React.

O frontend captura a aproximadamente 10 FPS, com apenas um frame pendente para evitar acúmulo. O token é revalidado durante conexões longas, a cada 30 segundos de frames; uma reconexão obtém o token atualizado do Supabase Auth.

Se o token utilizado na conexão expirar, o servidor fecha com código de reconexão (`1012`), permitindo buscar o token renovado sem recriar a sessão. Falhas temporárias de autenticação usam `1013`; rejeições definitivas de autorização usam `1008`.

Não foi corrigida deliberadamente a ausência de `eyesClosed` na saída com rosto do detector original, nem a contagem original de bocejos/piscadas na interface: isso alteraria comportamento de detecção fora do escopo.

## Sessões e limites operacionais

- Permite somente uma sessão aberta por câmera, usando verificação no banco e exclusão mútua no processo.
- Cada sessão mantém seu próprio FaceMesh, contadores e histórico, em uma thread dedicada.
- Cria a sessão após obter permissão da webcam. Se o socket não conectar/não enviar frames, encerra por timeout.
- Desconexão permite reconectar na mesma sessão e preservar o estado por até `RECONNECT_SECONDS` (30 segundos por padrão) desde o último frame. Depois disso é necessário iniciar uma nova sessão.
- Parar, sair da página, logout ou desabilitar a câmera limpa os recursos. Se a aba fechar sem conseguir chamar a API, o timeout encerra a sessão.
- **Execute apenas um processo/uma instância, sem múltiplos workers.** Não há constraint nova de unicidade no banco nem coordenação distribuída. Outro backend/cliente que escreva diretamente pode invalidar a garantia. Não use `--reload` em monitoramento real, pois reinícios reconciliam sessões.
- O vídeo continua dependendo da aba do navegador; não foi implementado RTSP nem gravação de vídeo.
- Não há fila durável de eventos: indisponibilidade do banco é registrada em log e enviada como mensagem de erro, sem parar vídeo/métricas; uma condição ainda ativa tenta persistir novamente no próximo frame. Quedas do servidor podem perder eventos não confirmados.

## Testar

Automatizado, a partir de `backend`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Frontend, a partir de `frontend`:

```powershell
npm run build
```

Os testes REST/WebSocket usam banco e Auth simulados; não validam credenciais nem RLS reais. O teste de paridade compara as saídas do código original e do adaptado ao longo de uma sequência com fadiga, bocejo, piscadas e rearme. Outro teste executa o MediaPipe real em uma imagem sem rosto.

Teste integrado manual:

1. Configurar um Supabase de teste e conferir colunas/tipos/RLS. Iniciar API e frontend.
2. Fazer login normalmente. Iniciar a câmera, permitir acesso e conferir vídeo processado, métricas e alertas.
3. Conferir que `sessions.user_id` é o usuário autenticado e que eventos aparecem com número/`metadata.metric` (ou no formato legado configurado).
4. Tentar iniciar a mesma câmera em outra aba: deve receber conflito sem nova sessão aberta. Negar a permissão da câmera não deve abrir sessão.
5. Consultar com outro usuário: câmeras compartilhadas visíveis, mas sessões/eventos privados. Tokens inválidos devem ser rejeitados.
6. Interromper a rede por menos de 30 segundos: deve reconectar na mesma sessão. Acima disso, a sessão deve receber `ended_at` e exigir novo início.
7. Parar pelo botão e fazer logout: não pode reconectar sozinho. Desabilitar a câmera pela API deve interromper o fluxo e impedir novo início.
8. Reiniciar a API com uma sessão abandonada: conferir reconciliação de `ended_at`.

O build do frontend passou. O bloqueio inicial do ESLint foi corrigido posteriormente a pedido do usuário: a configuração foi adaptada de TypeScript para JavaScript/JSX, foi instalado `eslint-plugin-react` como dependência de desenvolvimento para reconhecer o uso de componentes em JSX, e foram removidos imports, variáveis e parâmetros comprovadamente não utilizados. Foram modificados `frontend/eslint.config.js`, `frontend/package.json`, `frontend/package-lock.json`, `frontend/src/components/ui/badge.jsx`, `frontend/src/hooks/use-toast.js`, `frontend/src/pages/EmailResetPassword.jsx`, `frontend/src/pages/ResetPassword.jsx`, `frontend/src/pages/Login.jsx`, `frontend/src/pages/Monitoramento.jsx` e `frontend/src/services/auth.js`, sem alterar aparência ou regras de negócio.

Execute `npm run lint` em `frontend` para ver arquivo, linha e regra no terminal. Com a extensão ESLint no VS Code, os diagnósticos também aparecem no editor e no painel Problemas. O comando `npm run dev` não executa o lint automaticamente.

Resultado após a correção: lint concluído com **zero erros e três avisos** de Fast Refresh em `badge.jsx`, `button.jsx` e `sonner.jsx`, pois esses módulos compartilham componentes e outros exports. Os avisos foram mantidos visíveis, sem desativar regras ou reorganizar componentes fora do escopo. O build também passou novamente, com avisos não bloqueantes de Browserslist desatualizado e tamanho de bundle.

Durante a instalação da dependência de lint, o npm reportou 29 vulnerabilidades na árvore de dependências (6 moderadas e 23 altas). Não foi executado `npm audit fix` nem atualização ampla/forçada; a avaliação e correção dessas dependências é um trabalho separado.

Resultado automatizado: **12 testes aprovados**, incluindo o processamento real com MediaPipe e a comparação de paridade do detector. As bibliotecas emitem avisos não bloqueantes do MediaPipe e de depreciação do transporte HTTPX no TestClient do Starlette.
