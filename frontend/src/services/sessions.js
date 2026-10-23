import { api } from "./api";

// userId é ignorado intencionalmente: a API usa exclusivamente o JWT validado.
export const createSession = (_userId, deviceId) => api("/sessions", {
  method: "POST", body: { device_id: deviceId },
});

export const endSession = (sessionId) => api(`/sessions/${sessionId}/end`, { method: "POST" });
export const getSessions = () => api("/sessions");
export const getActiveSession = async () => (await api("/sessions?active=true"))[0] || null;
