from fastapi import Depends, HTTPException, Header, status
from typing import Optional, List
from config import get_settings
from DBmanager1 import UserDBHandler
userdbhandler = UserDBHandler()

settings = get_settings()

def get_current_user(x_user_email: Optional[str] = Header(None)):
    """
    Simulates user authentication. 
    In a real Azure AD production setup, this would validate the JWT Bearer token 
    using the Tenant ID and Client ID.
    
    For this setup, we expect the frontend or API Gateway to forward the verified email
    or we use a simple check.
    """
    if not x_user_email:
        # For development ease if header is missing, maybe default to admin? 
        # But user asked for production level auth... context implies strictness.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication header (X-User-Email)"
        )

   
        
    owner_email = userdbhandler.get_owner_email(x_user_email)
    if not owner_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Owner email not found please login first"
        )
    return x_user_email
