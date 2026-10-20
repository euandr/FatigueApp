import { supabase } from "./supabase";

/**
 * Retorna timestamp atual no fuso horário de Brasília (UTC-3)
 * @returns {string} ISO string com timezone de Brasília
 */
function getBrasiliaTimestamp() {
  const now = new Date();

  // Cria formatter para Brasília (UTC-3)
  const formatter = new Intl.DateTimeFormat("pt-BR", {
    timeZone: "America/Sao_Paulo",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });

  const parts = formatter.formatToParts(now);
  const partMap = {};

  parts.forEach(({ type, value }) => {
    partMap[type] = value;
  });

  // Constrói ISO string válida no horário de Brasília
  const isoString = `${partMap.year}-${partMap.month}-${partMap.day}T${partMap.hour}:${partMap.minute}:${partMap.second}`;

  return isoString;
}

/**
 * Salva um evento detectado na tabela events
 * @param {string} sessionId - ID da sessão (UUID)
 * @param {string} eventType - Descrição legível do evento (ex: 'Bocejo detectado')
 * @param {string} eventValue - Métrica do evento (ex: 'MAR: 0.752')
 * @returns {Promise<Object | null>}
 */
export async function saveEvent(sessionId, eventType, eventValue) {
  try {
    if (!sessionId || eventType === undefined || eventValue === undefined) {
      console.warn("❌ Parâmetros inválidos para saveEvent", {
        sessionId,
        eventType,
        eventValue,
      });
      return null;
    }

    console.log(`📤 Salvando ${eventType}: ${eventValue}`);

    const { data, error } = await supabase
      .from("events")
      .insert([
        {
          session_id: sessionId,
          event_type: eventType, // Descrição legível (ex: "Bocejo detectado")
          value: eventValue, // Métrica (ex: "MAR: 0.752")
          created_at: getBrasiliaTimestamp(),
        },
      ])
      .select()
      .single();

    if (error) {
      console.error(`❌ Erro Supabase ao salvar evento:`, {
        code: error.code,
        message: error.message,
      });
      throw error;
    }

    console.log(`✅ ${eventType} salvo`);
    return data;
  } catch (error) {
    console.error(`❌ Erro ao salvar evento:`, error.message);
    throw error;
  }
}

/**
 * Obtém todos os eventos de uma sessão
 * @param {string} sessionId - ID da sessão (UUID)
 * @returns {Promise<Array>}
 */
export async function getEventsBySessionId(sessionId) {
  try {
    if (!sessionId) {
      console.warn("Session ID não disponível");
      return [];
    }

    const { data, error } = await supabase
      .from("events")
      .select("*")
      .eq("session_id", sessionId)
      .order("created_at", { ascending: false });

    if (error) throw error;
    return data || [];
  } catch (error) {
    console.error("Erro ao buscar eventos:", error.message);
    return [];
  }
}

/**
 * Salva múltiplos eventos em lote (útil para batch de eventos recebidos)
 * @param {string} sessionId - ID da sessão
 * @param {Array} events - Array de eventos com {event_type, value}
 * @returns {Promise<Array>}
 */
export async function saveEventsBatch(sessionId, events) {
  try {
    if (!sessionId || !events.length) {
      return [];
    }

    const eventsData = events.map((event) => ({
      session_id: sessionId,
      event_type: event.event_type, // Descrição legível
      value: event.value, // Métrica
      created_at: event.created_at || getBrasiliaTimestamp(),
    }));

    const { data, error } = await supabase
      .from("events")
      .insert(eventsData)
      .select();

    if (error) throw error;
    console.log(`${data.length} eventos salvos em lote`);
    return data || [];
  } catch (error) {
    console.error("Erro ao salvar eventos em lote:", error.message);
    return [];
  }
}

/**
 * Delete all events for a session (útil para testes ou limpeza)
 * @param {string} sessionId - ID da sessão
 * @returns {Promise<Boolean>}
 */
export async function deleteSessionEvents(sessionId) {
  try {
    const { error } = await supabase
      .from("events")
      .delete()
      .eq("session_id", sessionId);

    if (error) throw error;
    return true;
  } catch (error) {
    console.error("Erro ao deletar eventos:", error.message);
    return false;
  }
}
