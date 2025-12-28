# Servidor de Detecção de Fadiga

Este é o backend Python para o Sistema de Detecção de Fadiga. Ele processa os frames de vídeo da webcam e retorna dados de detecção em tempo real.

## Requisitos

- Python 3.8 ou superior
- Webcam funcional

## Instalação

1. Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

```bash
python fatigue_server.py
```

O servidor irá iniciar em `ws://localhost:8765`.

## Como Funciona

1. O frontend React captura frames da webcam
2. Cada frame é enviado via WebSocket para este servidor
3. O servidor processa o frame usando MediaPipe Face Mesh
4. Detecta sinais de fadiga:
   - **EAR (Eye Aspect Ratio)**: Mede abertura dos olhos
   - **MAR (Mouth Aspect Ratio)**: Detecta bocejos
   - **Piscadas**: Conta frequência de piscadas
5. Retorna o frame anotado + dados de detecção

## Parâmetros de Detecção

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| EYE_AR_THRESH | 0.2 | Limiar EAR para olhos fechados |
| EYE_AR_CONSEC_FRAMES | 30 | Frames consecutivos para alerta de fadiga |
| MOUTH_AR_THRESH | 0.6 | Limiar MAR para bocejo |
| BLINK_CONSEC_FRAMES | 3 | Frames para detectar uma piscada |

## Estrutura dos Dados

### Entrada (frame do navegador):
```json
{
  "frame": "base64_encoded_jpeg"
}
```

### Saída (dados de detecção):
```json
{
  "ear": 0.28,
  "mar": 0.35,
  "blinks": 2,
  "totalBlinks": 15,
  "eyesClosed": false,
  "yawnDetected": false,
  "excessBlinks": false,
  "fatigueAlert": false,
  "frame": "base64_encoded_annotated_jpeg"
}
```

## Solução de Problemas

### Erro de MediaPipe
```
pip uninstall mediapipe
pip install mediapipe
```

### Erro de OpenCV
```
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Porta já em uso
Altere a porta no arquivo `fatigue_server.py`:
```python
port = 8766  # ou outra porta disponível
```

## Licença

MIT License
