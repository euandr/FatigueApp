import { api } from "./api";

// Apenas o detector no backend produz eventos; o navegador não grava alertas.
export const getEventsBySessionId = (sessionId) => api(`/sessions/${sessionId}/events`);
