from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
import os

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")

        if auth_header is None or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header missing or invalid")

        token = auth_header[len("Bearer "):]

        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get("userId")  # Assuming 'sub' stores user_id
            if user_id is None:
                raise HTTPException(status_code=401, detail="Invalid token: user_id missing")

            # Attach user_id to request.state so downstream endpoints can access it
            request.state.user_id = user_id

        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        response = await call_next(request)
        return response
