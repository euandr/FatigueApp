# 📹 Entendendo Sessões e Câmeras

## ❓ Resposta Rápida

**SIM**, toda vez que você clica "Iniciar Monitoramento", **cria uma NOVA SESSÃO** no BD.

**Sobre câmera**: Existe um campo `camera: "camera1"` nos **eventos**, mas **NÃO na sessão**. Veja a estrutura:

---

## 🗂️ Estrutura Atual do BD

### Tabela `sessions`
```
id (UUID)          → ID único da sessão
user_id (UUID)     → Qual usuário (não qual câmera!)
started_at         → Quando começou
ended_at           → Quando terminou
```

**❌ Problema**: Sem campo de câmera na sessão!

### Tabela `events`
```
id (UUID)          → ID único do evento
session_id (UUID)  → Qual sessão
event_type         → "Bocejo detectado"
value              → "yawn" (tipo técnico)
camera             → "camera1" ← Aqui tem a câmera!
```

**✅ Tem câmera nos eventos**, mas só dos eventos críticos, não de todas as métricas.

---

## 📊 Exemplo Prático

Digamos que você Usou a câmera 3 vezes em um dia:

```
├─ 09:00 - Clica "Iniciar"
│  └─ Cria Session #1
│  └─ Câmera ligada por 30 min
│  └─ Alguns eventos com camera: "camera1"
│  └─ Clica "Parar"
│  └─ Cria Event adicional se houver
│
├─ 14:00 - Clica "Iniciar" AGAIN
│  └─ Cria Session #2 (NOVA SESSÃO!)
│  └─ Mesma câmera (camera1) mas SESSION DIFERENTE
│  └─ Câmera ligada por 20 min
│  └─ Eventos vinculados a Session #2
│
└─ 20:00 - Clica "Iniciar" NOVAMENTE
   └─ Cria Session #3 (OUTRA NOVA SESSÃO!)
   └─ Eventos vinculados a Session #3
```

**No BD, teríamos:**
```
📋 SESSIONS
┌─────────────────────────────────────────┐
│ id        │ user_id │ started_at │ ... │
├─────────────────────────────────────────┤
│ sess-001  │ user-1  │ 09:00      │ ... │
│ sess-002  │ user-1  │ 14:00      │ ... │
│ sess-003  │ user-1  │ 20:00      │ ... │
└─────────────────────────────────────────┘
        ↑ 3 SESSÕES DIFERENTES! (mesma câmera, mesmo usuário)

📋 EVENTS (apenas eventos críticos)
┌─────────────────────────────────────────┐
│ session_id │ event_type │ camera │ ...  │
├─────────────────────────────────────────┤
│ sess-001   │ Bocejo     │ camera1│ ...  │
│ sess-001   │ Fadiga     │ camera1│ ...  │
│ sess-002   │ Bocejo     │ camera1│ ...  │
│ sess-003   │ Fadiga     │ camera1│ ...  │
└─────────────────────────────────────────┘
```

---

## 🎥 Suporte para Múltiplas Câmeras

Se você tiver **múltiplas câmeras** no futuro (ex: câmera frontal + câmera lateral):

### Solução 1: **Adicionar coluna `camera_id` na SESSÃO** (recomendado)

```sql
ALTER TABLE sessions ADD COLUMN camera_id VARCHAR(50) DEFAULT 'camera1';

-- Agora a sessão registra qual câmera foi usada
SELECT * FROM sessions;
-- id  | user_id | camera_id | started_at | ...
-- ... | ...     | camera1   | ...        | ...
```

**Fluxo atualizado:**
```javascript
export async function createSession(userId, cameraId = 'camera1') {
  const { data, error } = await supabase
    .from("sessions")
    .insert([
      {
        user_id: userId,
        camera_id: cameraId,  // ← Novo!
        started_at: new Date().toISOString(),
      },
    ])
    .select()
    .single();

  if (error) throw error;
  return data;
}
```

### Solução 2: **Tabela separada `cameras_devices`** (futuro/avançado)

Se você quiser rastrear QUAL device físico (ex: "Meu Notebook", "Servidor"):

```sql
CREATE TABLE cameras (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  device_id VARCHAR(100),        -- Identificador único do device
  device_name VARCHAR(100),      -- "Meu Notebook", "Servidor Lab"
  location VARCHAR(100),         -- "sala 1", "escritório"
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Adicionar camera_id como FK
ALTER TABLE sessions ADD COLUMN camera_id UUID REFERENCES cameras(id);
```

Mas por enquanto **NÃO recomendo** (adiciona complexidade).

---

## 🔍 Como Saber "É a Mesma Câmera"?

Atualmente:

```javascript
// ❌ NÃO RECOMENDADO (pode não funcionar)
// Comparando eventos diferentes de sessões diferentes
const session1Events = await getEventsBySessionId('sess-001');
const session2Events = await getEventsBySessionId('sess-002');

// Como saber se é mesma câmera?
// Resposta: Não há como garantir sem campo na sessão!
```

**✅ Solução: Adicionar campo `camera_id` na sessão**

```javascript
// Fácil comparar depois:
const session1 = await getSession('sess-001');
const session2 = await getSession('sess-002');

if (session1.camera_id === session2.camera_id) {
  console.log('Mesma câmera!');
}
```

---

## 📋 Estrutura Proposta (Melhorada)

### Tabela `sessions` (Atualizada)
```sql
CREATE TABLE sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id),
  camera_id VARCHAR(50) DEFAULT 'camera1',  -- ← NOVO!
  started_at TIMESTAMPTZ,
  ended_at TIMESTAMPTZ,
  notes TEXT                                -- Opcional: "Cansado", "Testando"
);
```

### Tabela `events` (Sem mudanças)
```sql
CREATE TABLE events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id),
  event_type VARCHAR(100),
  value VARCHAR(50),
  camera VARCHAR(50),  -- "camera1", "camera2", etc
  created_at TIMESTAMPTZ
);
```

### Tabela `metrics_timeseries` (Nova)
```sql
CREATE TABLE metrics_timeseries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES sessions(id),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  ear DECIMAL(4,3),
  mar DECIMAL(4,3),
  blinks_in_interval INT,
  -- ... métricas
);
```

---

## 🎯 Fluxo Atualizado no Hook

### Código Atual (lado do JavaScript)

```javascript
// useFatigueDetection.js

const startStreaming = useCallback(async () => {
  // ... (pegar câmera via getUserMedia)
  
  // ✨ Passa cameraId para createSession
  const session = await createSession(userId, 'camera1');
  setSessionId(session.id);
  
  // ... resto do código
}, []);
```

### Código Atualizado em `sessions.js`

```javascript
export async function createSession(userId, cameraId = 'camera1') {
  try {
    const { data, error } = await supabase
      .from("sessions")
      .insert([
        {
          user_id: userId,
          camera_id: cameraId,  // ← Novo parâmetro
          started_at: new Date().toISOString(),
        },
      ])
      .select()
      .single();

    if (error) throw error;
    return data;
  } catch (error) {
    console.error("Erro ao criar sessão:", error.message);
    throw error;
  }
}

// Nova função: Verificar ligações entre múltiplas sessões
export async function getSessionsByCamera(userId, cameraId) {
  try {
    const { data, error } = await supabase
      .from("sessions")
      .select("*")
      .eq("user_id", userId)
      .eq("camera_id", cameraId)
      .order("started_at", { ascending: false });

    if (error) throw error;
    return data || [];
  } catch (error) {
    console.error("Erro ao buscar sessões por câmera:", error.message);
    return [];
  }
}

// Nova função: Comparar dados de múltiplas sessões
export async function compareSessionsByCamera(userId, cameraId) {
  const sessions = await getSessionsByCamera(userId, cameraId);
  
  // Retorna todas as sessões usadas com essa câmera
  return sessions.map(session => ({
    id: session.id,
    started_at: session.started_at,
    ended_at: session.ended_at,
    camera_id: session.camera_id,
  }));
}
```

---

## 🤔 Cenários Comuns

### Cenário 1: "Quero saber todos os dias que usei a câmera frontal"

```javascript
const sessions = await getSessionsByCamera(userId, 'camera1');
// Retorna array com todas as sessões usando essa câmera
```

### Cenário 2: "Quero comparar fadiga entre 2 sessões diferentes"

```javascript
const metrics1 = await getMetricsForSession('sess-001');
const metrics2 = await getMetricsForSession('sess-002');

// Calcular média de EAR em cada sessão
const avgEAR1 = metrics1.reduce((sum, m) => sum + m.ear, 0) / metrics1.length;
const avgEAR2 = metrics2.reduce((sum, m) => sum + m.ear, 0) / metrics2.length;

console.log(`Sessão 1 (EAR médio): ${avgEAR1}`);
console.log(`Sessão 2 (EAR médio): ${avgEAR2}`);
```

### Cenário 3: "Tenho 2 câmeras diferentes, como registro?"

```javascript
// Quando user clica em "Câmera Frontal"
await startStreaming('camera1');

// Quando user clica em "Câmera Lateral"
await startStreaming('camera2');

// Cada uma cria uma SESSION DIFERENTE com camera_id diferente
```

---

## ✅ Checklist da Implementação

- [ ] Entender que **cada "Iniciar" = NOVA SESSÃO**
- [ ] Adicionar coluna `camera_id` na tabela `sessions`
- [ ] Atualizar função `createSession()` para aceitar `cameraId`
- [ ] Atualizar hook `useFatigueDetection.js` para passar câmera
- [ ] Criar função `getSessionsByCamera()` para análise
- [ ] Testar com múltiplas ativações da câmera

---

## 🎥 Detalhes Técnicos: Como o Sistema Identifica Câmera?

**Método 1: Hard-coded** (atual)
```javascript
camera: "camera1"  // Sempre "camera1"
```
Funciona se você tem UMA câmera. Simples.

**Método 2: Device ID** (futuro)
```javascript
const stream = await navigator.mediaDevices.getUserMedia({video: true});
const videoTrack = stream.getVideoTracks()[0];
const deviceId = videoTrack.getSettings().deviceId;
// deviceId é algo como "e8f7c326b1c..."

camera: deviceId  // ID único gerado pelo SO
```
Funciona mesmo com múltiplas câmeras, mas é mais complexo.

**Método 3: User selection** (recomendado)
```javascript
// Deixar user escolher
<select onChange={(e) => setCameraId(e.target.value)}>
  <option value="camera1">Câmera Frontal</option>
  <option value="camera2">Câmera Lateral</option>
</select>
```

---

## 📞 Resumo Final

| Pergunta | Resposta |
|----------|----------|
| **Toda vez que ativo a câmera, é nova sessão?** | ✅ SIM |
| **Há info da câmera na DB?** | ⚠️ SIM, mas só nos eventos (não na sessão) |
| **Como saber se é a mesma câmera?** | 🔲 Atual: não há; Proposto: adicionar `camera_id` |
| **E se tiver 2 câmeras?** | 🎯 Usar campo `camera_id` para diferenciar |

**ação imediata**: Adicionar `camera_id` na tabela `sessions`. Deixe-me saber se quer que eu implemente! 
