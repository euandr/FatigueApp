# FatigueDetector - Resumo Técnico do Projeto

## 📋 Visão Geral

**FatigueDetector** é uma aplicação React + Vite que detecta sinais de fadiga em tempo real através de análise de vídeo. Integra processamento de visão computacional (backend em Python/WebSocket) com um frontend web para monitoramento, alertas e armazenamento de dados.

---

## 🏗️ Arquitetura do Sistema

### Três Camadas Principais

```
┌─────────────────────────────────────────┐
│         User Interface (React)           │
│  - Login, Cadastro, Monitoramento       │
│  - Video Feed, Métricas, Eventos        │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│    Lógica de Detecção & Dados            │
│  - useFatigueDetection hook              │
│  - Integração Supabase (Auth, BD)        │
│  - Cache de eventos e sessões            │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│  Backend Externo (WebSocket)             │
│  - Servidor Python na porta 8765         │
│  - Processamento de frames (visão CV)    │
│  - Retorna métricas de fadiga em tempo real
└─────────────────────────────────────────┘
```

---

## 🔑 Componentes Principais

### 1️⃣ **Hook de Detecção** (`src/hooks/useFatigueDetection.js`)

**Coração da aplicação**

- Conecta via WebSocket ao servidor de detecção (tipo: `wss://` em produção, `ws://localhost:8765` em dev)
- Captura frames da câmera via `getUserMedia`
- Envia frames para o servidor via WebSocket
- Recebe métricas em tempo real:
  - **EAR** (Eye Aspect Ratio): medida de abertura dos olhos
  - **MAR** (Mouth Aspect Ratio): medida de abertura da boca (para bociejo)
  - **Flags**: `eyesClosed`, `yawnDetected`, `excessBlinks`, `fatigueAlert`
- Toca alarmes quando fadiga é detectada
- Cria/encerra sessões de monitoramento
- Salva eventos no Supabase

**Estado gerenciado:**

```javascript
{
  isStreaming, isConnected, isMuted, sessionId, userId,
  processedFrame, events, yawnCount,
  metrics: { ear, mar, blinks, eyesClosed, yawnDetected, ... }
}
```

### 2️⃣ **Autenticação & Banco de Dados** (`src/lib/`)

- `supabase.js`: Inicializa cliente Supabase (Auth + Database)
- `auth.js`: Gerencia login, cadastro, reset de senha
- `sessions.js`: Cria sessões de monitoramento, registra timestamps
- `events.js`: Salva eventos de fadiga (bociejo, piscar excessivo, etc.)

**Fluxo de dados:**

- Usuário faz login → obtém `userId` do Supabase Auth
- Inicia monitoramento → `createSession()` cria registro no BD
- Eventos de fadiga → `saveEvent()` armazena no Supabase
- Encerra monitoramento → `endSession()` finaliza sessão com timestamp

### 3️⃣ **Interface React** (`src/pages/` e `src/components/`)

**Páginas:**

- `Login.jsx`: Autenticação com email/senha
- `Cadastro.jsx`: Registro de novo usuário
- `Monitoramento.jsx`: Dashboard principal com vídeo + métricas
- `ResetPassword.jsx` / `EmailResetPassword.jsx`: Recuperação de senha
- `NotFound.jsx`: Página 404

**Componentes:**

- `VideoFeed.jsx`: Exibe stream de vídeo e canvas procesado
- `ControlPanel.jsx`: Botões de start/stop, silenciar alarme
- `MetricsPanel.jsx`: Exibe EAR, MAR, contadores em tempo real
- `EventsPanel.jsx`: Histórico de eventos (bociejo, piscar, etc.)
- `ui/*`: Componentes Shadcn/UI (Card, Button, Input, Badge, etc.)

---

## 📊 Fluxo de Dados Completo

```
1. Usuário faz Login
   └─> Supabase Auth valida credenciais
   └─> userId salvo no estado

2. Clica "Iniciar Monitoramento"
   └─> createSession() cria registro no BD
   └─> useFatigueDetection.js obtém câmera

3. Câmera transmite frames
   └─> VideoFeed exibe via <video> tag
   └─> Hook envia frame via WebSocket

4. Backend Python processa
   └─> Calcula EAR, MAR, detecta bociejo, piscar
   └─> Retorna métricas via WebSocket

5. Hook recebe métricas
   └─> MetricsPanel atualiza UI em tempo real
   └─> Se fatigar detectada:
     ├─> playAlarm() toca som (ou silenciado)
     └─> addEvent() salva no Supabase

6. EventsPanel mostra histórico
   └─> Exibe cardápios de eventos recentes

7. Usuário para monitoramento
   └─> endSession() finaliza sessão no BD
   └─> Fecha WebSocket
```

---

## 🔄 Fluxo de Autenticação

```
Não autenticado
   ├─> Login.jsx (email + senha)
   ├─> Cadastro.jsx (novo usuário)
   └─> ResetPassword.jsx (recuperar senha)
         │
         ↓
Autenticado
   └─> Monitoramento.jsx (dashboard)
         └─> VideoFeed + Métricas + Eventos
```

---

## 🛠️ Stack Técnico

| Camada          | Tecnologia                          |
| --------------- | ----------------------------------- |
| **Frontend**    | React 18 + Vite                     |
| **UI Lib**      | Shadcn/UI + Tailwind CSS            |
| **API/DB**      | Supabase (Auth + PostgreSQL)        |
| **Websocket**   | Native WebSocket API                |
| **HTTP Client** | Fetch API                           |
| **Forms**       | React Hook Form                     |
| **Requests**    | React Query (@tanstack/react-query) |
| **Styling**     | Tailwind CSS + PostCSS              |

---

## 📦 Variáveis de Ambiente

```env
# .env ou .env.local
VITE_SUPABASE_URL=<url-do-supabase>
VITE_SUPABASE_KEY=<chave-publica>
VITE_WS_URL=<wss://seu-servidor.com:8765> (ou deixe vazio para localhost)
```

---

## 🎯 Próximas Fases

### Fase 1: Consolidação de Dados ✏️ (Seu foco atual)

- ✅ Detecção de fadiga funciona
- 🔲 Estruturar tabelas no Supabase para relatórios
- 🔲 Criar queries para análise de dados
- 🔲 Implementar exportação de dados

### Fase 2: Relatórios & Analytics

- Gráficos de fadiga ao longo do tempo
- Estatísticas por sessão (média de EAR, quantidade de bociejo, etc.)
- Exportar em PDF/CSV

### Fase 3: Melhorias UX

- Dashboard com histórico de sessões
- Alertas personalizáveis
- Sugestões de descanso baseadas em padrões

---

## 📁 Estrutura de Arquivos

```
src/
├── pages/              # Páginas React (rotas)
│   ├── Login.jsx
│   ├── Cadastro.jsx
│   ├── Monitoramento.jsx          ← Dashboard principal
│   ├── ResetPassword.jsx
│   └── NotFound.jsx
├── components/         # Componentes reutilizáveis
│   ├── VideoFeed.jsx               ← Exibe vídeo
│   ├── ControlPanel.jsx            ← Controles
│   ├── MetricsPanel.jsx            ← Métricas em tempo real
│   ├── EventsPanel.jsx             ← Histórico de eventos
│   └── ui/                         # Shadcn/UI components
├── hooks/             # Custom React hooks
│   └── useFatigueDetection.js      ← ❤️ CORE
├── lib/               # Lógica & APIs
│   ├── supabase.js                 ← Inicialização DB
│   ├── auth.js                     ← Login/Cadastro
│   ├── sessions.js                 ← Criar/encerrar sessões
│   ├── events.js                   ← Salvar eventos
│   └── utils.js                    ← Helpers
└── assets/            # Imagens, áudio
    └── alarm-clock.mp3            ← Som de alarme
```

---

## 🔗 Integrações

### Supabase

- **Auth**: Login com email/senha
- **Database**: Armazena sessões, eventos, métricas
- **RLS** (Row Level Security): Cada usuário vê apenas seus dados

### Backend Externo (Python/OpenCV)

- Roda em `localhost:8765` (dev) ou `wss://seu-servidor.com:8765` (produção)
- Comunica via WebSocket
- Recebe: frames em base64 ou binary
- Retorna: JSON com métricas de fadiga

---

## ✅ Checklist de Funcionalidade

- ✅ WebSocket connection establecida
- ✅ Detecção de EAR/MAR em tempo real
- ✅ Alertas/alarmes funcionando
- ✅ Autenticação Supabase
- ✅ Salvamento de eventos básico
- 🔲 Queries para relatórios
- 🔲 Exportação de dados
- 🔲 Dashboard com análises

---

## 🚀 Como Rodar

```bash
# Instalar dependências
npm install

# Rodas dev (com hot reload)
npm run dev

# Build para produção
npm run build

# Lint
npm lint
```

---

## 📞 Dúvidas Frequentes

**P: Como o WebSocket se reconecta se cair?**
R: `useFatigueDetection.js` tem lógica de retry automático com backoff exponencial.

**P: Onde estão as tabelas do Supabase?**
R: Criadas via dashboard Supabase. Tabelas principais: `sessions`, `events`, `users` (gerenciada pelo Auth).

**P: Pode usar em produção?**
R: Sim, depois de: (1) testar com usuários reais, (2) configurar CORS, (3) certificados SSL para WSS.
