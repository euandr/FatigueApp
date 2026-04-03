# 📊 Análise de Métricas - Arquitetura para Gráficos

## 📡 O Que o Python Retorna (WebSocket)

Cada frame (~10 FPS = a cada 100ms), o servidor Python envia um JSON com essa estrutura:

```json
{
  "ear": 0.35, // Eye Aspect Ratio (0.0 - 0.5)
  "mar": 0.15, // Mouth Aspect Ratio (0.0 - 0.3)
  "blinks": 2, // Número de piscadas NO ÚLTIMO INTERVALO
  "totalBlinks": 47, // Total acumulado de piscadas na sessão
  "eyesClosed": false, // Boolean: olhos fechados?
  "yawnDetected": false, // Boolean: bocejo detectado?
  "excessBlinks": false, // Boolean: piscadas em excesso?
  "fatigueAlert": false, // Boolean: alerta de fadiga?
  "frame": "base64_encoded_image" // Imagem processada (opcional)
}
```

---

## 📈 Métricas Explicadas

| Métrica                      | Range      | O Que É                               | Para Quê                             |
| ---------------------------- | ---------- | ------------------------------------- | ------------------------------------ |
| **EAR** (Eye Aspect Ratio)   | 0.0 - 0.5  | Proporção: altura olho / largura olho | Detectar olhos fechados/abertos      |
| **MAR** (Mouth Aspect Ratio) | 0.0 - 0.3  | Proporção: altura boca / largura boca | Detectar bocejos                     |
| **blinks**                   | 0 - 5+     | Piscadas no intervalo atual (~100ms)  | Taxa de piscadas                     |
| **totalBlinks**              | 0 - ∞      | Total acumulado                       | Piscadas totais na sessão            |
| **Flags booleanas**          | true/false | Detecções binárias                    | Eventos críticos (trigger de alarme) |

### Interpretação de Valores

```
EAR (Eye Aspect Ratio):
├─ EAR < 0.15-0.2   → Olhos FECHADOS (risco de fadiga)
├─ EAR > 0.25-0.3   → Olhos ABERTOS (normal)
└─ EAR gradualmente diminuindo → Fadiga progredindo

MAR (Mouth Aspect Ratio):
├─ MAR > 0.15       → Boca ABERTA (possível bocejo)
└─ MAR < 0.1        → Boca FECHADA (normal)

blinks:
├─ 0-1 a cada 100ms → Normal (~6-10 por minuto)
├─ 2-3 a cada 100ms → Aumentado (~12-18 por minuto)
└─ >3 a cada 100ms  → EXCESSO (possível fadiga)

totalBlinks:
├─ ~12 por minuto    → Normal
└─ >20 por minuto    → Anormal (fadiga)
```

---

## 🗄️ Estrutura de Tabelas Recomendada

### Opção 1: **Métricas Contínuas** (RECOMENDADO para gráficos)

Armazena EVERY frame recebido = granularidade máxima.

```sql
CREATE TABLE metrics_timeseries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id),
  timestamp TIMESTAMPTZ DEFAULT NOW(),

  -- Métricas do frame atual
  ear DECIMAL(4,3),           -- 0.0 - 0.5
  mar DECIMAL(4,3),           -- 0.0 - 0.3
  blinks_in_interval INT,     -- 0-5
  total_blinks INT,           -- 0+

  -- Flags de detecção
  eyes_closed BOOLEAN,
  yawn_detected BOOLEAN,
  excess_blinks BOOLEAN,
  fatigue_alert BOOLEAN,

  -- Para rastreamento
  frame_number INT,           -- Qual frame foi este

  CONSTRAINT metrics_timeseries_session_fk FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Índices para queries rápidas
CREATE INDEX idx_metrics_session_time ON metrics_timeseries(session_id, timestamp);
CREATE INDEX idx_metrics_ear ON metrics_timeseries(session_id, ear);
CREATE INDEX idx_metrics_mar ON metrics_timeseries(session_id, mar);
```

**Prós:**

- ✅ Gráficos suave e detalhado (séries de tempo)
- ✅ Análises estatísticas (média, desvio padrão, etc.)
- ✅ Detectar padrões ao longo da sessão

**Contras:**

- ⚠️ Mais dados armazenados (~60 linhas/minuto = 3600/hora)
- ⚠️ Queries podem ser mais lentas em sessões longas

---

### Opção 2: **Agregação por Intervalos** (alternativa com menos storage)

Armazena médias/resumos a cada 10 segundos (ou 1 minuto).

```sql
CREATE TABLE metrics_aggregated (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id),
  interval_start TIMESTAMPTZ,
  interval_end TIMESTAMPTZ,

  -- Estatísticas do intervalo
  ear_avg DECIMAL(4,3),
  ear_min DECIMAL(4,3),
  ear_max DECIMAL(4,3),

  mar_avg DECIMAL(4,3),
  mar_min DECIMAL(4,3),
  mar_max DECIMAL(4,3),

  blinks_total INT,
  blinks_per_minute DECIMAL(5,2),

  -- Contadores de eventos no intervalo
  eyes_closed_count INT,
  yawn_count INT,
  excess_blinks_count INT,
  fatigue_alerts_count INT,

  -- Sentimento geral do intervalo
  fatigue_risk_level VARCHAR(10), -- 'low', 'medium', 'high'

  CONSTRAINT agg_session_fk FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX idx_aggregated_session_time ON metrics_aggregated(session_id, interval_start);
```

**Prós:**

- ✅ Menos storage (1 linha a cada 10 segundos)
- ✅ Queries mais rápidas
- ✅ Ainda permite gráficos

**Contras:**

- ⚠️ Menos granularidade
- ⚠️ Perde informações detalhadas

---

## ✅ Recomendação: Estratégia Híbrida

**Use AMBAS as tabelas:**

1. **`metrics_timeseries`**: Armazena dados brutos (todos os frames)
   - Para análises detalhadas
   - Manter por ~7 dias apenas (depois arquivar/deletar)
2. **`metrics_aggregated`**: Armazena resumos a cada 10-60 segundos
   - Manter por tempo indefinido (pouco storage)
   - Para gráficos históricos de longo prazo

```
Fluxo:
Frontend recebe frame do Python
  ├─ INSERT em metrics_timeseries (imediato)
  ├─ A cada 10 segundos, calcula agregação
  └─ INSERT em metrics_aggregated

[7 dias depois]
  └─ Deleta dados antigos de metrics_timeseries
  └─ Mantém agregações em metrics_aggregated
```

---

## 📊 Casos de Uso para Gráficos

### Gráfico 1: **EAR ao longo do tempo** (Fadiga Progredindo)

```sql
SELECT timestamp, ear
FROM metrics_timeseries
WHERE session_id = $1
ORDER BY timestamp;
```

→ Mostra se usuário está ficando mais cansado (EAR diminuindo)

### Gráfico 2: **MAR ao longo do tempo** (Bocejos)

```sql
SELECT timestamp, mar
FROM metrics_timeseries
WHERE session_id = $1
ORDER BY timestamp;
```

→ Picos = bocejos

### Gráfico 3: **Taxa de Piscadas por Minuto**

```sql
SELECT
  DATE_TRUNC('minute', timestamp) as minute,
  SUM(blinks_in_interval) as blinks
FROM metrics_timeseries
WHERE session_id = $1
GROUP BY minute
ORDER BY minute;
```

→ Mostra variação na taxa de piscadas ao longo da sessão

### Gráfico 4: **Risco de Fadiga ao Longo do Tempo** (Heatmap)

```sql
SELECT
  timestamp,
  CASE
    WHEN ear < 0.15 THEN 'high'
    WHEN ear < 0.25 THEN 'medium'
    ELSE 'low'
  END as risk_level
FROM metrics_timeseries
WHERE session_id = $1
ORDER BY timestamp;
```

### Gráfico 5: **Resumo por Sessão** (Dashboard)

```sql
SELECT
  s.id,
  s.started_at,
  s.ended_at,
  COUNT(m.id) as total_frames,
  AVG(m.ear) as avg_ear,
  MIN(m.ear) as min_ear,
  COUNT(CASE WHEN m.yawn_detected THEN 1 END) as yawn_count,
  COUNT(CASE WHEN m.eyes_closed THEN 1 END) as eyes_closed_frames
FROM sessions s
LEFT JOIN metrics_timeseries m ON s.id = m.session_id
WHERE s.user_id = $1
GROUP BY s.id
ORDER BY s.started_at DESC;
```

---

## 🔧 Implementação: Como Salvar Métricas

### Passo 1: Atualizar o Hook (`useFatigueDetection.js`)

```javascript
// Adicionar função para salvar métrica contínua
const saveMetric = useCallback(async (metrics) => {
  if (!sessionIdRef.current) return;

  try {
    const { error } = await supabase.from("metrics_timeseries").insert({
      session_id: sessionIdRef.current,
      ear: metrics.ear,
      mar: metrics.mar,
      blinks_in_interval: metrics.blinks,
      total_blinks: metrics.totalBlinks,
      eyes_closed: metrics.eyesClosed,
      yawn_detected: metrics.yawnDetected,
      excess_blinks: metrics.excessBlinks,
      fatigue_alert: metrics.fatigueAlert,
    });

    if (error) throw error;
  } catch (error) {
    console.error("Erro ao salvar métrica:", error);
  }
}, []);

// No onmessage do WebSocket, chamar:
wsRef.current.onmessage = (event) => {
  const data = JSON.parse(event.data);
  setMetrics(data);

  // ✨ NOVO: Salvar métrica a cada frame
  saveMetric(data);

  // ... resto do código
};
```

**Ou para economizar storage**, salvar a cada 10 frames (1 segundo):

```javascript
let frameCount = 0;

wsRef.current.onmessage = (event) => {
  const data = JSON.parse(event.data);
  setMetrics(data);

  // Salvar a cada 10 frames (~1 segundo)
  if (++frameCount % 10 === 0) {
    saveMetric(data);
  }

  // ... resto
};
```

### Passo 2: Criar Tabelas no Supabase

```sql
-- Rodar no SQL Editor do Supabase

-- Tabela de métricas contínuas
CREATE TABLE metrics_timeseries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW(),

  ear DECIMAL(4,3),
  mar DECIMAL(4,3),
  blinks_in_interval INT,
  total_blinks INT,

  eyes_closed BOOLEAN,
  yawn_detected BOOLEAN,
  excess_blinks BOOLEAN,
  fatigue_alert BOOLEAN
);

CREATE INDEX idx_metrics_session_time ON metrics_timeseries(session_id, created_at);

-- Habilitar RLS
ALTER TABLE metrics_timeseries ENABLE ROW LEVEL SECURITY;

-- Política: usuários só veem métricas de suas próprias sessões
CREATE POLICY metrics_user_isolation ON metrics_timeseries FOR SELECT
  USING (
    session_id IN (
      SELECT id FROM sessions WHERE user_id = auth.uid()
    )
  );
```

### Passo 3: Criar Queries no `lib/`

Criar novo arquivo: `src/lib/metrics.js`

```javascript
import { supabase } from "./supabase";

export async function saveMetric(sessionId, metricData) {
  const { error } = await supabase.from("metrics_timeseries").insert({
    session_id: sessionId,
    ear: metricData.ear,
    mar: metricData.mar,
    blinks_in_interval: metricData.blinks,
    total_blinks: metricData.totalBlinks,
    eyes_closed: metricData.eyesClosed,
    yawn_detected: metricData.yawnDetected,
    excess_blinks: metricData.excessBlinks,
    fatigue_alert: metricData.fatigueAlert,
  });

  if (error) throw error;
}

export async function getMetricsForSession(sessionId) {
  const { data, error } = await supabase
    .from("metrics_timeseries")
    .select("*")
    .eq("session_id", sessionId)
    .order("created_at", { ascending: true });

  if (error) throw error;
  return data;
}

export async function getMetricsAggregated(sessionId, intervalSeconds = 60) {
  // Query que agrupa métricas em intervalos
  const { data, error } = await supabase.rpc("get_metrics_aggregated", {
    session_id: sessionId,
    interval_seconds: intervalSeconds,
  });

  if (error) throw error;
  return data;
}
```

---

## 📱 Estrutura de Página para Gráficos

```jsx
// pages/Relatorios.jsx (novo)
import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { getMetricsForSession } from "@/lib/metrics";

export default function Relatorios() {
  const [metrics, setMetrics] = useState([]);
  const sessionId = new URLSearchParams(window.location.search).get(
    "session_id",
  );

  useEffect(() => {
    getMetricsForSession(sessionId).then(setMetrics);
  }, [sessionId]);

  return (
    <div>
      <h1>Relatório da Sessão</h1>

      {/* Gráfico EAR */}
      <LineChart data={metrics} width={600} height={300}>
        <CartesianGrid />
        <XAxis dataKey="created_at" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="ear" stroke="#8884d8" />
      </LineChart>

      {/* Gráfico MAR */}
      <LineChart data={metrics} width={600} height={300}>
        <CartesianGrid />
        <XAxis dataKey="created_at" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="mar" stroke="#82ca9d" />
      </LineChart>
    </div>
  );
}
```

---

## 📋 Resumo: O Que Fazer

| Passo | O Quê                             | Arquivo                            |
| ----- | --------------------------------- | ---------------------------------- |
| 1     | Criar tabelas no Supabase         | Dashboard Supabase (SQL Editor)    |
| 2     | Criar função para salvar métricas | `src/lib/metrics.js` (novo)        |
| 3     | Atualizar hook para salvar        | `src/hooks/useFatigueDetection.js` |
| 4     | Criar página de gráficos          | `src/pages/Relatorios.jsx` (novo)  |
| 5     | Instalar biblioteca charts        | `npm install recharts`             |
| 6     | Construir gráficos                | Usar Recharts ou Chart.js          |

---

## 🎯 Próxima Pergunta?

Quer que eu:

- ✅ **Implemente** a tabela e funções?
- ✅ **Crie as queries** SQL detalhadas?
- ✅ **Mostre exemplos de gráficos** com React + Recharts?
- ✅ **Explique mais** sobre alguma métrica específica?
