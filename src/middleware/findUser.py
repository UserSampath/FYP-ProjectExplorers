from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.exception import raise_custom_error

from jose import jwt, JWTError
import os

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise_custom_error(401, "You are not authorized")
        return user_id
    except JWTError:
        raise_custom_error(401, "You are not authorized")
