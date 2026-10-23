from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

bearer = HTTPBearer(auto_error=False)


async def authenticated_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
):
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Informe o token Bearer do Supabase")
    return await request.app.state.db.user(credentials.credentials)
