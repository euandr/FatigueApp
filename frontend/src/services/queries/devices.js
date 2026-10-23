import { api } from "@/services/api";

// As assinaturas são preservadas; a identidade é validada pela API, não pelo userId.
export async function createOrGetDevice(_userId, deviceName = null) {
  try {
    return { data: await api("/devices", {
      method: "POST", body: { name: deviceName || "CAM_01" },
    }) };
  } catch (error) {
    return { error };
  }
}

export async function getActiveDevices() {
  try {
    return { data: await api("/devices?active=true") };
  } catch (error) {
    return { error, data: [] };
  }
}

export async function getDevice(deviceId) {
  try {
    return { data: await api(`/devices/${deviceId}`) };
  } catch (error) {
    return { error };
  }
}

export async function updateDevice(deviceId, updates) {
  try {
    return { data: await api(`/devices/${deviceId}`, { method: "PATCH", body: updates }) };
  } catch (error) {
    return { error };
  }
}

export const deactivateDevice = (deviceId) => updateDevice(deviceId, { is_active: false });
export const activateDevice = (deviceId) => updateDevice(deviceId, { is_active: true });
