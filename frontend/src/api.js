// import axios from "axios";

// const api = axios.create({
//   baseURL: "http://localhost:8000",
// });

import axios from "axios";

export const API_URL = (
  import.meta.env.VITE_API_URL ||
  `${window.location.protocol}//${window.location.hostname}:8000`
).replace(/\/$/, "");

export const WS_URL =
  import.meta.env.VITE_WS_URL || `${API_URL.replace(/^http/, "ws")}/ws`;

const api = axios.create({
  baseURL: API_URL,
});
// Camera API


export async function getCameras(){
     const response = await api.get("/cameras");
     return response.data;
}
export async function getCameraById(cameraId) {
  const response = await api.get(`/cameras/${cameraId}`);
  return response.data;
}

export async function postCamera(camera) {
  const response = await api.post("/cameras/registrar", camera);
  return response.data;
}

export async function patchCamera(cameraId, dadosAtualizados) {
  const response = await api.patch(`/cameras/atualizar/${cameraId}`, dadosAtualizados);
  return response.data;
}

export async function deleteCamera(cameraId) {
  const response = await api.delete(`/cameras/deletar/${cameraId}`);
  return response.data;
}