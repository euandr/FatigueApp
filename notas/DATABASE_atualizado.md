# Banco de dados — Projeto de Detecção de Fadiga

## 1. Visão geral

O banco possui **três tabelas principais** no schema `public`, além da autenticação de usuários fornecida pelo Supabase Auth (schema `auth`):

- `devices`: fontes de vídeo/câmeras cadastradas no sistema, sem vínculo direto com um usuário.
- `sessions`: períodos de monitoramento de cada câmera.
- `events`: ocorrências de fadiga detectadas durante uma sessão.

O sistema utiliza **Supabase Auth** para autenticação. A referência ao usuário é feita por `auth.users.id`.

### Relacionamentos

```text

auth.users (id)

    └──< sessions (user_id)

devices (id)

    └──< sessions (device_id)

             └──< events (session_id)

```

- As câmeras são recursos cadastrados no sistema, sem proprietário individual definido na tabela `devices`.
- Uma câmera pode ter várias sessões ao longo do tempo.
- Cada sessão pertence a uma câmera e pode ter vários eventos.
- `sessions.user_id` representa o usuário autenticado que iniciou o monitoramento. A tabela `devices` não possui `user_id`. O backend deve validar a autorização por regras próprias, ainda pendentes de definição.

## 2. Tabela `devices`

**Finalidade:** representar uma fonte de vídeo cadastrada. O registro é persistente mesmo que a câmera se desconecte ou seja desabilitada.

| Coluna | Tipo PostgreSQL | Nulidade | Descrição |

|---|---|---|---|

| `id` | `uuid` | `NOT NULL` | Chave primária; identificador persistente da câmera. |

| `name` | `text` | `NOT NULL` | Nome amigável, ex.: `Sala 01`. |

| `created_at` | `timestamptz` | `NOT NULL` | Data de cadastro. |

| `is_active` | `boolean` | `NOT NULL` | **Habilitada/desabilitada pelo responsável**, não indica conectividade. |

**Campos deliberadamente adiados:**

- `source_type` | `text` | **A confirmar** | Tipo de fonte, por exemplo `browser`, `usb`, `rtsp`. Campo **planejado**, confirmar se já foi criado. |
- `source_config` (`jsonb`): configuração específica da fonte, especialmente câmeras IP/RTSP. Não adicionar sem necessidade; nunca expor credenciais RTSP ao frontend.
- `last_seen_at` (`timestamptz`): última comunicação confirmada; opcional para auditoria e diagnóstico.
- `is_connected`: **não criar inicialmente**. O estado de conexão deve ser calculado pelo backend a partir do fluxo de vídeo, WebSocket e/ou timeout.

### Significado de `is_active`

- `true`: câmera **habilitada** e autorizada a ser monitorada.
- `false`: câmera **desabilitada** pelo operador autorizado; o backend não deve permitir iniciar monitoramento nela e deve interromper o monitoramento ativo se for desabilitada.

Uma câmera pode estar **habilitada e desconectada** simultaneamente. Conectividade e habilitação são conceitos diferentes.

## 3. Tabela `sessions`

**Finalidade:** representar cada período de monitoramento iniciado para uma câmera.

| Coluna | Tipo PostgreSQL | Nulidade | Descrição |

|---|---|---|---|

| `id` | `uuid` | `NOT NULL` | Chave primária da sessão. |

| `user_id` | `uuid` | `NOT NULL` | FK para `auth.users(id)`; usuário que iniciou a sessão. |

| `device_id` | `uuid` | `NOT NULL` | FK para `devices(id)`; câmera monitorada. |

| `started_at` | `timestamptz` | `NOT NULL` | Início da sessão. |

| `ended_at` | `timestamptz` | `NULL` permitido | Encerramento da sessão. |

**Decisão:** **não adicionar `status` agora**. A versão inicial usa:

- `ended_at IS NULL`: sessão **registrada como aberta**, mas isso **não prova** que o processamento está ativo.
- `ended_at IS NOT NULL`: sessão encerrada.

Consequências e regras de implementação:

1. Criar uma sessão quando o monitoramento for iniciado com sucesso.
2. Preencher `ended_at` quando o responsável encerrar o monitoramento.
3. Impedir mais de uma sessão aberta para o mesmo `device_id`.
4. Se a câmera perder conexão, aplicar uma política de reconexão; **a política exata ainda precisa ser definida**.
5. Se o backend cair, sessões podem permanecer abertas indevidamente. Na inicialização, verificar e reconciliar sessões abandonadas antes de aceitar novos monitoramentos.
6. Sem `status`, não é possível distinguir historicamente encerramento normal de interrupção. Se isso passar a ser requisito, adicionar posteriormente `status` ou `end_reason`.

## 4. Tabela `events`

**Finalidade:** registrar eventos relevantes de fadiga detectados pelo processamento, associados a uma sessão.

| Coluna | Tipo PostgreSQL | Nulidade | Descrição |

|---|---|---|---|

| `id` | `uuid` | `NOT NULL` | Chave primária do evento. |

| `session_id` | `uuid` | `NOT NULL` | FK para `sessions(id)`. |

| `event_type` | `text` | `NOT NULL` | Descrição ou código do evento detectado. |

| `value` | `numeric` | **Preferencialmente `NOT NULL` para os três eventos atuais** | Valor numérico da métrica. **Alteração planejada:** antes era `text`. |

| `metadata` | `jsonb` | `NOT NULL` | Metadados do evento, contendo pelo menos a chave `metric`. **Alteração planejada:** confirmar se já foi criada. |

| `created_at` | `timestamptz` | `NOT NULL` | Instante da detecção. |

### Os três eventos atualmente detectados

| `event_type` atual | `value` antigo (`text`) | `value` novo (`numeric`) | `metadata` |

|---|---|---:|---|

| `Bocejo detectado` | `MAR: 0.692` | `0.692` | `{"metric":"MAR"}` |

| `Excesso de piscadas detectado (5 em 30 frames)` | `Blinks: 5` | `5` | `{"metric":"blinks"}` |

| `Alerta de fadiga detectado - olhos fechados por muito tempo` | `EAR: 0.078` | `0.078` | `{"metric":"EAR"}` |

Os valores acima são **exemplos**. O texto exato de `event_type` deve ser conferido no código atual, pois pode conter números dinâmicos ou diferenças de grafia.

metadata pode conter outros valores alem desse.

### Exemplo de registro desejado

```json

{

  "session_id": "<uuid-da-sessao>",

  "event_type": "Bocejo detectado",

  "value": 0.692,

  "metadata": {

    "metric": "MAR"

  }

}

```

**Observações importantes:**

- `event_type` descreve **o que ocorreu**; `metadata.metric` identifica **qual métrica foi registrada em `value`**.
- O significado de `value` varia conforme a métrica: MAR, quantidade de piscadas ou EAR. Não comparar valores de métricas diferentes como se tivessem a mesma unidade.
- O texto `5 em 30 frames` contém o contexto da janela de detecção. Se esse parâmetro puder variar, futuramente registrar também `window_frames` em `metadata` ou em uma configuração versionada; caso contrário, os dados históricos podem ficar ambíguos.
- O EAR isolado não registra por quanto tempo os olhos ficaram fechados. Para estudos posteriores, considerar guardar duração ou frames consecutivos.
- Não gravar EAR/MAR a cada frame em `events` por padrão; essa tabela é para **ocorrências**, não telemetria contínua.
- Evitar eventos duplicados a cada frame enquanto uma condição persistir; definir início/limiar/rearme dos detectores.

### Restrição opcional de `metadata`

`NOT NULL` **não** impede `metadata = {}`. Caso seja necessário exigir a propriedade `metric` no banco:

## 5. Responsabilidades do FastAPI

O backend deve:

1. **Autenticar** requisições usando o JWT do Supabase Auth e validar o usuário responsável.
2. **Autorizar** operações em `devices`, `sessions` e `events` conforme a propriedade da câmera/sessão.
3. Gerenciar cadastro, habilitação e desabilitação de câmeras.
4. Iniciar e encerrar sessões, garantindo associação correta a `device_id` e `user_id`.
5. Receber ou capturar frames, conforme o adaptador de vídeo utilizado.
6. Processar **cada câmera com estado de detecção independente** (contadores, EAR/MAR, frames consecutivos etc.).
7. Persistir **eventos detectados** no Supabase e transmitir métricas/alertas ao React em tempo real.
8. Manter o estado transitório de conexão em memória inicialmente; não confundir conexão com `devices.is_active`.
9. Tratar desconexões, timeouts, reconexões e reinicializações, sem deixar sessões abertas indefinidamente.
10. Nunca disponibilizar a chave `service_role` do Supabase no React ou em código cliente.

**Importante:** o FastAPI pode usar credenciais de backend para gravar dados, mas isso não dispensa validação explícita de identidade e autorização. Se usar `service_role`, ela pode contornar RLS: a aplicação deverá garantir a autorização em cada operação. Definir e testar políticas RLS adequadas para acessos diretos do frontend.

## 6. Arquitetura de vídeo e monitoramento

### Durante os testes

- Notebook/celular: o navegador captura frames JPEG e os envia por WebSocket, aproximadamente 10 FPS, como no protótipo atual.
- O navegador é uma **fonte provisória** de frames; não é a arquitetura definitiva de captura.

### Para implantação futura

- Câmeras IP provavelmente usarão RTSP ou outra interface de streaming suportada.
- O processamento ocorrerá no servidor Python, com módulos de captura intercambiáveis.
- O React servirá como painel de controle e visualização; o backend deverá conseguir continuar o monitoramento quando o painel for fechado, **desde que a fonte de vídeo não dependa da aba do navegador**.
- O responsável autenticado decide quando iniciar e interromper o monitoramento; não foi definido funcionamento obrigatório 24 horas por dia.
- O vídeo será **analisado em tempo real**, sem gravação de arquivos de vídeo inicialmente.

**Ainda pendente:** definir se o servidor ficará na rede local das câmeras ou na nuvem; essa decisão afeta o acesso a câmeras IP em redes privadas.

## 7. Regras de acesso

- não ha niveis de usuario ainda.
- o usuário autorizado pode cadastrar, visualizar, habilitar/desabilitar e iniciar/parar monitoramento de suas câmeras, bem como consultar suas sessões e eventos.
- O usuário monitorado pela câmera **não** é necessariamente o usuário do sistema (`user_id`).
- Não há compartilhamento entre múltiplos usuários definido para esta versão.
- Políticas de RLS devem refletir a cadeia de acesso: `events -> sessions -> devices -> user_id`.

## 9. Orientação para o OpenCode

**Antes de implementar:** inspecione o código atual do frontend, detector Python, funções de persistência e esquema real do Supabase. Trate as tabelas e decisões acima como **modelo-alvo documentado**, não como prova de que todas as migrações já foram executadas. Proponha alterações incrementais e peça confirmação antes de mudanças destrutivas no banco.
