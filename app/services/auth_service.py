import httpx
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.config import AppConfig, logger

class AuthService:
    """
    Handles authentication with the remote backend, including login and token refreshing.
    """
    _instance: Optional["AuthService"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._access_token: Optional[str] = AppConfig.API_TOKEN
        self._refresh_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._base_url = AppConfig.API_URL
        
        # Load tokens from environment/config if available
        # In a production app, we might want to store these in a more persistent local DB
        pass

    def get_access_token(self) -> Optional[str]:
        """Returns the current access token."""
        return self._access_token

    async def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticates with the remote backend using username and password.
        Updates internal token state on success.
        """
        url = f"{self._base_url}/account/token"
        logger.info(f"[Auth] Attempting login to {url} for user {username}")
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url, 
                    json={"username": username, "password": password},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self._update_tokens(data.get("access"), data.get("refresh"))
                    logger.info("[Auth] Login successful")
                    return {"success": True, "message": "Login successful"}
                else:
                    logger.warning(f"[Auth] Login failed: {response.status_code} - {response.text}")
                    return {"success": False, "message": "Invalid credentials"}
            except Exception as e:
                logger.error(f"[Auth] Login error: {e}")
                return {"success": False, "message": f"Connection error: {str(e)}"}

    async def refresh_access_token(self) -> bool:
        """
        Uses the refresh token to get a new access token.
        """
        if not self._refresh_token:
            logger.warning("[Auth] No refresh token available")
            return False
            
        url = f"{self._base_url}/account/token/refresh"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url, 
                    json={"refresh": self._refresh_token},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    new_access = data.get("access")
                    new_refresh = data.get("refresh") or self._refresh_token
                    
                    self._update_tokens(new_access, new_refresh)
                    logger.info("[Auth] Token refresh successful")
                    return True
                else:
                    logger.warning(f"[Auth] Token refresh failed: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"[Auth] Token refresh error: {e}")
                return False

    def _update_tokens(self, access: str, refresh: str):
        """Updates the internal token state."""
        self._access_token = access
        self._refresh_token = refresh
        

        if access:
            from app.services.api_client import api_client
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(api_client.update_token(access))
            except RuntimeError:
                pass
    
    def is_authenticated(self) -> bool:
        return self._access_token is not None

    def logout(self):
        """Logs out the user by clearing the tokens."""
        self._access_token = None
        self._refresh_token = None
        
        from app.services.api_client import api_client
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(api_client.update_token(None))
        except RuntimeError:
            pass
            
        logger.info("[Auth] User logged out")

auth_service = AuthService()
