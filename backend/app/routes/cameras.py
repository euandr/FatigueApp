from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ..supabase import supabase

router = APIRouter(
    prefix="/cameras",
    tags=["Cameras"]
)


class CriarCamera(BaseModel):
     nome: str 
     protocolo: str
     url: str
class AtualizarCamera(BaseModel):
     name: str | None = None
     protocolo: str | None = None
     stream_url: str | None = None

@router.get("/")
def listar_cameras():
     response = supabase.table("devices").select("*").execute()
     return response.data

@router.post("/registrar")
def registrar_camera(CriarCamera: CriarCamera):
     try:
          response = (
               supabase.table("devices")
               .insert({"name": CriarCamera.nome, "protocolo": CriarCamera.protocolo, "stream_url": CriarCamera.url})
               .execute()
          )
          return {"message": "Camera registrada com sucesso", "data": response.data}
     except Exception as e:
          return {"error": str(e)}

@router.patch("/atualizar/{camera_id}")
def atualizar_camera(camera_id: str, atualizarCamera: AtualizarCamera):
     dados_tratados = atualizarCamera.model_dump(exclude_none=True)

     response = (
          supabase.table("devices")
          .update(dados_tratados)
          .eq("id", camera_id)
          .execute()
     )
     if not response.data:
        raise ValueError("Transação não encontrada")
     return {"message": "ok", "data": response.data[0]}

@router.delete("/deletar/{camera_id}")
def deletar_camera(camera_id: str):
     response =(
          supabase.table("devices")
          .delete()
          .eq("id", camera_id)
          .execute()
     )
     if not response.data:
        raise ValueError("Transação não encontrada")
     return {"message": "ok", "data": response.data[0]}