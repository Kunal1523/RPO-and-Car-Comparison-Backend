from fastapi import FastAPI,APIRouter, HTTPException,Depends,BackgroundTasks
import requests
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode
import time
import httpx
import pdb
from dotenv import load_dotenv

load_dotenv()


CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID")
CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET")
REDIRECT_URI = os.getenv("MICROSOFT_REDIRECT_URI")
TENANT_ID = os.getenv("TENANT_ID")
SCOPE = os.getenv("MICROSOFT_SCOPE")
# AUTH_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/authorize"
# TOKEN_URL = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
AUTH_URL = f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
TOKEN_URL = f"https://login.microsoftonline.com/common/oauth2/v2.0/token"
USERINFO_URL = "https://graph.microsoft.com/v1.0/me"
MICROSOFT_GRAPH_URL = "https://graph.microsoft.com/v1.0/me/sendMail"








async def refresh_microsoft_token(refresh_token: str) -> dict:
    pdb.set_trace()
    token_url = TOKEN_URL
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
        "scope": SCOPE
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to refresh Microsoft token")
        print(response.json())
        return response.json()
    

async def refresh_access_token(refresh_token: str):
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "scope": SCOPE,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(TOKEN_URL, data=data)

    if resp.status_code != 200:
        raise Exception("Failed to refresh Microsoft access token")

    token_data = resp.json()

    return {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token", refresh_token),
        "expires_at": datetime.now(timezone.utc)
        + timedelta(seconds=token_data["expires_in"]),
    }
