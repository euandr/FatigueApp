import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
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