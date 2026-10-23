"""Acesso ao PostgREST com credencial privada, sem SDK adicional."""
import logging

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, config):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.supabase_url,
            headers={"apikey": config.service_key,
                     "Authorization": f"Bearer {config.service_key}"},
            timeout=15,
        )

    async def close(self):
        await self.client.aclose()

    async def user(self, token):
        # O Auth do Supabase verifica assinatura, expiração e identidade do JWT.
        # Não confiar em claims apenas decodificadas pelo cliente.
        try:
            response = await self.client.get(
                "/auth/v1/user", headers={"Authorization": f"Bearer {token}"}
            )
        except httpx.HTTPError:
            raise HTTPException(503, "Serviço de autenticação indisponível")
        if response.status_code in (401, 403):
            raise HTTPException(401, "Token inválido ou expirado")
        if response.is_error:
            raise HTTPException(503, "Não foi possível validar a autenticação")
        user = response.json()
        if not user.get("id"):
            raise HTTPException(401, "Usuário inválido")
        return user

    async def query(self, table, method="GET", params=None, data=None):
        try:
            response = await self.client.request(
                method, f"/rest/v1/{table}", params=params, json=data,
                headers={"Prefer": "return=representation"},
            )
        except httpx.HTTPError:
            raise HTTPException(503, "Banco de dados indisponível")
        if response.is_error:
            logger.error("Falha no banco: %s %s (%s)", method, table, response.status_code)
            raise HTTPException(502, "Operação no banco falhou; confira o esquema e a configuração do backend")
        return response.json() if response.content else []

    async def check_schema(self):
        # Consultas somente de leitura; nenhum DDL ou migração automática.
        columns = {
            "devices": "id,name,created_at,is_active",
            "sessions": "id,user_id,device_id,started_at,ended_at",
            "events": "id,session_id,event_type,value,created_at" + (
                ",metadata" if self.config.event_schema == "numeric" else ""
            ),
        }
        for table, select in columns.items():
            await self.query(table, params={"select": select, "limit": "0"})
