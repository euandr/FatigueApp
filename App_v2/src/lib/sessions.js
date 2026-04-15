import { supabase } from "./supabase";

/**
 * Cria uma nova sessão de detecção de fadiga
 * @param {string} userId - ID do usuário (UUID)
 * @param {string} deviceId - ID do device/câmera (UUID)
 * @returns {Promise<{id: string, user_id: string, device_id: string, started_at: string} | null>}
 */
export async function createSession(userId, deviceId) {
  try {
    if (!deviceId) {
      throw new Error("deviceId é obrigatório para criar uma sessão");
    }

    const { data, error } = await supabase
      .from("sessions")
      .insert([
        {
          user_id: userId,
          device_id: deviceId,
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

/**
 * Finaliza uma sessão de detecção
 * @param {string} sessionId - ID da sessão (UUID)
 * @returns {Promise<{id: string, user_id: string, ended_at: string} | null>}
 */
export async function endSession(sessionId) {
  try {
    const { data, error } = await supabase
      .from("sessions")
      .update({
        ended_at: new Date().toISOString(),
      })
      .eq("id", sessionId)
      .select()
      .single();

    if (error) throw error;
    return data;
  } catch (error) {
    console.error("Erro ao finalizar sessão:", error.message);
    throw error;
  }
}

/**
 * Obtém todas as sessões de um usuário
 * @param {string} userId - ID do usuário (UUID)
 * @returns {Promise<Array>}
 */
export async function getSessions(userId) {
  try {
    const { data, error } = await supabase
      .from("sessions")
      .select("*")
      .eq("user_id", userId)
      .order("started_at", { ascending: false });

    if (error) throw error;
    return data || [];
  } catch (error) {
    console.error("Erro ao buscar sessões:", error.message);
    throw error;
  }
}

/**
 * Obtém a sessão ativa (sem ended_at) de um usuário
 * @param {string} userId - ID do usuário (UUID)
 * @returns {Promise<Object | null>}
 */
export async function getActiveSession(userId) {
  try {
    const { data, error } = await supabase
      .from("sessions")
      .select("*")
      .eq("user_id", userId)
      .is("ended_at", null)
      .order("started_at", { ascending: false })
      .limit(1)
      .single();

    if (error && error.code !== "PGRST116") throw error; // PGRST116 = no rows found
    return data || null;
  } catch (error) {
    console.error("Erro ao buscar sessão ativa:", error.message);
    return null;
  }
}
