import { supabase } from "@/lib/supabase";

/**
 * Registra/cria um novo device para o usuário
 * Se um device com o mesmo name já existe, retorna o existing
 * @param {string} userId - ID do usuário
 * @param {string} deviceName - Nome do device (opcional, usa default do BD)
 * @returns {Promise<Object>} Device criado/existente ou erro
 */
export async function createOrGetDevice(userId, deviceName = null) {
  try {
    // Se há um nome específico, tenta encontrar device existente com esse nome
    if (deviceName) {
      const { data: existing, error: fetchError } = await supabase
        .from("devices")
        .select("*")
        .eq("user_id", userId)
        .eq("name", deviceName)
        .eq("is_active", true)
        .single();

      if (!fetchError && existing) {
        return { data: existing, isNew: false };
      }
    }

    // Cria novo device com name default do BD ou o especificado
    const deviceData = {
      user_id: userId,
      is_active: true,
    };

    if (deviceName) {
      deviceData.name = deviceName;
    }

    const { data, error } = await supabase
      .from("devices")
      .insert([deviceData])
      .select()
      .single();

    if (error) throw error;
    return { data, isNew: true };
  } catch (error) {
    console.error("Error creating/getting device:", error);
    return { error };
  }
}

/**
 * Busca todos os devices ativos do usuário
 * @param {string} userId - ID do usuário
 * @returns {Promise<Array>} Lista de devices ativos
 */
export async function getActiveDevices(userId) {
  try {
    const { data, error } = await supabase
      .from("devices")
      .select("*")
      .eq("user_id", userId)
      .eq("is_active", true)
      .order("created_at", { ascending: false });

    if (error) throw error;
    return { data };
  } catch (error) {
    console.error("Error fetching active devices:", error);
    return { error, data: [] };
  }
}

/**
 * Busca um device específico por ID
 * @param {string} deviceId - ID do device
 * @returns {Promise<Object>} Device encontrado ou erro
 */
export async function getDevice(deviceId) {
  try {
    const { data, error } = await supabase
      .from("devices")
      .select("*")
      .eq("id", deviceId)
      .single();

    if (error) throw error;
    return { data };
  } catch (error) {
    console.error("Error fetching device:", error);
    return { error };
  }
}

/**
 * Atualiza um device
 * @param {string} deviceId - ID do device
 * @param {Object} updates - Campos a atualizar
 * @returns {Promise<Object>} Device atualizado ou erro
 */
export async function updateDevice(deviceId, updates) {
  try {
    const { data, error } = await supabase
      .from("devices")
      .update(updates)
      .eq("id", deviceId)
      .select()
      .single();

    if (error) throw error;
    return { data };
  } catch (error) {
    console.error("Error updating device:", error);
    return { error };
  }
}

/**
 * Desativa um device (soft delete)
 * @param {string} deviceId - ID do device
 * @returns {Promise<Object>} Device desativado ou erro
 */
export async function deactivateDevice(deviceId) {
  return updateDevice(deviceId, { is_active: false });
}

/**
 * Ativa um device desativado
 * @param {string} deviceId - ID do device
 * @returns {Promise<Object>} Device ativado ou erro
 */
export async function activateDevice(deviceId) {
  return updateDevice(deviceId, { is_active: true });
}
