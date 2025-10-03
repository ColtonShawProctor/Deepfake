from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.auth import authenticate_user, create_access_token, get_current_user, create_user, get_password_hash
from app.models.user import User
from app.schemas import UserCreate, UserResponse, Token, AuthResponse
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["authentication"])

def is_valid_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_password(password: str) -> bool:
    """Simple password validation"""
    return len(password) >= 8

class TestLoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register", response_model=AuthResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Validate email format
    if not is_valid_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )
    
    # Validate password strength
    if not is_valid_password(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    try:
        # Create new user
        user = create_user(db, user_data.email, user_data.password)
        
        # Create access token
        token_data = create_access_token(data={"sub": str(user.id)})
        
        return AuthResponse(
            success=True,
            message="User registered successfully",
            access_token=token_data,
            token_type="bearer",
            expires_in=24 * 60 * 60,  # 24 hours in seconds
            user=UserResponse(
                id=user.id,
                email=user.email,
                created_at=user.created_at
            )
        )
        
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.get("/test-model")
async def test_model(db: Session = Depends(get_db)):
    """Test endpoint to check if User model is working in API context"""
    try:
        # Try to query the database
        user_count = db.query(User).count()
        print(f"DEBUG: User count in API context: {user_count}")
        
        # Try to find a specific user
        user = db.query(User).filter(User.email == "test@test.com").first()
        if user:
            print(f"DEBUG: User found in API context: {user.email}")
            return {"success": True, "user_count": user_count, "test_user_found": True}
        else:
            print("DEBUG: Test user not found in API context")
            return {"success": False, "user_count": user_count, "test_user_found": False}
            
    except Exception as e:
        print(f"DEBUG: Error in test-model endpoint: {e}")
        return {"success": False, "error": str(e)}

@router.post("/test-login")
async def test_login(request: TestLoginRequest, db: Session = Depends(get_db)):
    """Test login endpoint for debugging"""
    print(f"DEBUG: test_login called with email='{request.email}', password='{request.password}'")
    
    # Authenticate user
    user = authenticate_user(db, request.email, request.password)
    
    if not user:
        return {"success": False, "message": "Authentication failed"}
    
    return {"success": True, "message": "Authentication successful", "user_id": user.id}

@router.post("/login", response_model=AuthResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login user and return JWT token"""
    # Authenticate user
    user = authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Create access token
    token_data = create_access_token(data={"sub": str(user.id)})
    
    return AuthResponse(
        success=True,
        message="Login successful",
        access_token=token_data,
        token_type="bearer",
        expires_in=24 * 60 * 60,  # 24 hours in seconds
        user=UserResponse(
            id=user.id,
            email=user.email,
            created_at=user.created_at
        )
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        created_at=current_user.created_at
    )
