import { supabase } from "./supabase";

export const API_URL = (
  import.meta.env.VITE_API_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`
).replace(/\/$/, "");

export const WS_URL = import.meta.env.VITE_WS_URL ||
  `${API_URL.replace(/^http/, "ws")}/ws`;

export async function accessToken() {
  const { data, error } = await supabase.auth.getSession();
  if (error || !data.session?.access_token) {
    throw new Error("Usuário não autenticado");
  }
  return data.session.access_token;
}

export async function api(path, { method = "GET", body } = {}) {
  const token = await accessToken();
  const response = await fetch(`${API_URL}/api${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : "Falha na comunicação com a API");
  }
  return data;
}
