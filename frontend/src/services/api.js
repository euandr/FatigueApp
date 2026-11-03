export const API_URL = (
  import.meta.env.VITE_API_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`
).replace(/\/$/, "");

export const WS_URL =
  import.meta.env.VITE_WS_URL || `${API_URL.replace(/^http/, "ws")}/ws`;
