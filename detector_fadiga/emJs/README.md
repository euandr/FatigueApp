# Servidor de Detecção de Fadiga - Versão JavaScript

Conversão do servidor Python `serve_local.py` para JavaScript/Node.js usando WebSocket.

## Instalação

### Pré-requisitos

- Node.js >= 16.0.0
- npm

### Configuração

1. Instale as dependências:

```bash
npm install
```

2. (Opcional) Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

## Uso

Inicie o servidor:

```bash
npm start
```

O servidor será iniciado em `ws://0.0.0.0:8765` por padrão.

### Variáveis de Ambiente

- `WS_HOST`: Host para escutar (padrão: `0.0.0.0`)
  - Use `0.0.0.0` para aceitar conexões externas (produção)
  - Use `localhost` para apenas conexões locais (desenvolvimento)
- `WS_PORT`: Porta do servidor (padrão: `8765`)

## Funcionalidades

### Detecção de Fadiga

O servidor processa frames de vídeo enviados via WebSocket e retorna:

- **EAR (Eye Aspect Ratio)**: Métrica dos olhos abertos/fechados
- **MAR (Mouth Aspect Ratio)**: Métrica do bocejo
- **Blinks**: Número de piscadas na janela de detecção
- **Total Blinks**: Total de piscadas desde o início
- **Yawn Detected**: Detecção de bocejo
- **Excess Blinks**: Alerta de excesso de piscadas
- **Fatigue Alert**: Alerta de fadiga (olhos fechados por muito tempo)

### Constantes de Detecção

```javascript
EYE_AR_THRESH = 0.2; // Limiar de EAR para olhos fechados
EYE_AR_CONSEC_FRAMES = 30; // Frames consecutivos para alerta de fadiga
BLINK_CONSEC_FRAMES = 3; // Frames consecutivos para contar piscada
MOUTH_AR_THRESH = 0.6; // Limiar de MAR para bocejo
EXCESS_BLINKS_THRESH = 5; // Limite de piscadas para alerta
```

## Protocolo WebSocket

### Requisição

Envie um JSON com a frame em base64:

```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

### Resposta

O servidor retorna os dados de detecção:

```json
{
  "ear": 0.25,
  "mar": 0.15,
  "blinks": 2,
  "totalBlinks": 45,
  "yawnDetected": false,
  "excessBlinks": false,
  "fatigueAlert": false,
  "frame": "data:image/jpeg;base64,..."
}
```

## Dependências

- **ws**: WebSocket server para Node.js
- **@mediapipe/tasks-vision**: MediaPipe Face Landmarker
- **sharp**: Processamento de imagens
- **dotenv**: Gerenciamento de variáveis de ambiente
- **pino**: Logger estruturado

## Logs

O servidor usa Pino para logging estruturado. Os logs são exibidos no console em tempo real.

## Parar o Servidor

Pressione `Ctrl+C` para encerrar o servidor graceful.

## Notas

- A conversão mantém toda a lógica e funcionalidade da versão Python
- MediaPipe é carregado via CDN (requer conexão com internet)
- Ideal para desenvolvimento local e testes
- Para produção, considere usar a versão Python ou containerizar com Docker
