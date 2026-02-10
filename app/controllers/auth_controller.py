from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel
from app.services.auth_service import auth_service
from app.config import AppConfig

router = APIRouter(tags=["Authentication"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/api/login", summary="Login to the platform")
async def login(request: LoginRequest):
    result = await auth_service.login(request.username, request.password)
    
    if result["success"]:
        return JSONResponse(content={"message": "Login successful"})
    else:
        raise HTTPException(status_code=401, detail=result["message"])

@router.post("/api/logout", summary="Logout from the platform")
async def logout():
    auth_service.logout()
    return JSONResponse(content={"message": "Logout successful"})

@router.get("/api/auth/status", summary="Check authentication status")
async def check_auth_status():
    return {"authenticated": auth_service.is_authenticated()}

@router.get("/login", summary="Login Page")
async def get_login_page():
    return FileResponse(AppConfig.BASE_DIR / "login.html", media_type="text/html")
