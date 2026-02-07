"""
Authentication module with MongoDB storage and Gamification
"""
from datetime import datetime, timedelta
import os
from typing import Any, Optional
import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from bson import ObjectId

# Configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
SECRET_KEY = os.getenv("JWT_SECRET", "codeflow-ai-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# XP and Level configuration
XP_PER_LEVEL = 100
XP_ACTIONS = {
    "analyze_code": 10,
    "run_code": 5,
    "ask_ai": 8,
    "solve_problem": 25,
    "daily_login": 15,
    "streak_bonus": 5,  # Per day of streak
}

ACHIEVEMENTS = {
    "first_analysis": {"name": "First Steps", "desc": "Analyze your first code", "icon": "🚀", "xp": 50},
    "ten_analyses": {"name": "Code Explorer", "desc": "Analyze 10 codes", "icon": "🔍", "xp": 100},
    "fifty_analyses": {"name": "Analysis Master", "desc": "Analyze 50 codes", "icon": "🏆", "xp": 250},
    "first_run": {"name": "Code Runner", "desc": "Run your first code", "icon": "▶️", "xp": 30},
    "hundred_runs": {"name": "Execution Expert", "desc": "Run 100 codes", "icon": "⚡", "xp": 200},
    "ai_friend": {"name": "AI Friend", "desc": "Ask AI 20 questions", "icon": "🤖", "xp": 150},
    "streak_3": {"name": "On Fire", "desc": "3 day coding streak", "icon": "🔥", "xp": 75},
    "streak_7": {"name": "Week Warrior", "desc": "7 day coding streak", "icon": "💪", "xp": 150},
    "streak_30": {"name": "Monthly Master", "desc": "30 day coding streak", "icon": "👑", "xp": 500},
    "level_5": {"name": "Rising Star", "desc": "Reach level 5", "icon": "⭐", "xp": 100},
    "level_10": {"name": "Code Ninja", "desc": "Reach level 10", "icon": "🥷", "xp": 200},
    "level_25": {"name": "Legendary", "desc": "Reach level 25", "icon": "🏅", "xp": 500},
    "night_owl": {"name": "Night Owl", "desc": "Code after midnight", "icon": "🦉", "xp": 50},
    "early_bird": {"name": "Early Bird", "desc": "Code before 6 AM", "icon": "🐦", "xp": 50},
    "polyglot": {"name": "Polyglot", "desc": "Use 5 different languages", "icon": "🌍", "xp": 150},
}

# MongoDB client singleton
class DatabaseManager:
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None
    
    @classmethod
    async def get_database(cls) -> AsyncIOMotorDatabase:
        if cls._client is None:
            cls._client = AsyncIOMotorClient(MONGO_URL)
            cls._db = cls._client.codeflow_ai
        return cls._db  # type: ignore

async def get_database() -> AsyncIOMotorDatabase:
    return await DatabaseManager.get_database()

# Models
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: str

class TokenResponse(BaseModel):
    token: str
    user: UserResponse

# Router
router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_token(user_id: str) -> str:
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        database = await get_database()
        user = await database.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc

@router.post("/signup", response_model=dict)
async def signup(user: UserCreate):
    database = await get_database()
    
    # Check if user exists
    existing = await database.users.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user with gamification data
    user_doc = {
        "name": user.name,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.utcnow(),
        "problems_solved": 0,
        "streak": 0,
        "bookmarks": [],
        "history": [],
        # Gamification fields
        "xp": 0,
        "level": 1,
        "total_analyses": 0,
        "total_runs": 0,
        "total_questions": 0,
        "achievements": [],
        "languages_used": [],
        "last_active": datetime.utcnow(),
        "streak_updated": datetime.utcnow().date().isoformat(),
    }
    
    result = await database.users.insert_one(user_doc)
    return {"message": "Account created successfully", "id": str(result.inserted_id)}

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin):
    database = await get_database()
    
    # Find user
    db_user = await database.users.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Verify password
    if not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Create token
    token = create_token(str(db_user["_id"]))
    
    return TokenResponse(
        token=token,
        user=UserResponse(
            id=str(db_user["_id"]),
            name=db_user["name"],
            email=db_user["email"],
            created_at=db_user["created_at"].isoformat()
        )
    )

@router.get("/me", response_model=UserResponse)
async def get_me(user = Depends(get_current_user)):
    return UserResponse(
        id=str(user["_id"]),
        name=user["name"],
        email=user["email"],
        created_at=user["created_at"].isoformat()
    )

@router.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}


# ============================================================================
# Gamification Endpoints
# ============================================================================

class XPAction(BaseModel):
    action: str
    language: Optional[str] = None

class StatsResponse(BaseModel):
    xp: int
    level: int
    xp_to_next_level: int
    xp_progress_percent: int
    streak: int
    total_analyses: int
    total_runs: int
    total_questions: int
    achievements: list
    languages_used: list
    rank: Optional[int] = None

def calculate_level(xp: int) -> int:
    """Calculate level from XP."""
    return (xp // XP_PER_LEVEL) + 1

def xp_for_level(level: int) -> int:
    """Calculate XP needed for a specific level."""
    return (level - 1) * XP_PER_LEVEL

def xp_to_next_level(xp: int) -> int:
    """Calculate XP needed for next level."""
    current_level = calculate_level(xp)
    next_level_xp = current_level * XP_PER_LEVEL
    return next_level_xp - xp

async def check_and_award_achievements(database, user_id: str, user_data: dict) -> list:
    """Check for new achievements and award them."""
    new_achievements = []
    current_achievements = user_data.get("achievements", [])
    
    checks = [
        ("first_analysis", user_data.get("total_analyses", 0) >= 1),
        ("ten_analyses", user_data.get("total_analyses", 0) >= 10),
        ("fifty_analyses", user_data.get("total_analyses", 0) >= 50),
        ("first_run", user_data.get("total_runs", 0) >= 1),
        ("hundred_runs", user_data.get("total_runs", 0) >= 100),
        ("ai_friend", user_data.get("total_questions", 0) >= 20),
        ("streak_3", user_data.get("streak", 0) >= 3),
        ("streak_7", user_data.get("streak", 0) >= 7),
        ("streak_30", user_data.get("streak", 0) >= 30),
        ("level_5", user_data.get("level", 1) >= 5),
        ("level_10", user_data.get("level", 1) >= 10),
        ("level_25", user_data.get("level", 1) >= 25),
        ("polyglot", len(user_data.get("languages_used", [])) >= 5),
    ]
    
    # Check time-based achievements
    current_hour = datetime.utcnow().hour
    if current_hour >= 0 and current_hour < 5:
        checks.append(("night_owl", True))
    if current_hour >= 4 and current_hour < 6:
        checks.append(("early_bird", True))
    
    for achievement_id, condition in checks:
        if condition and achievement_id not in current_achievements:
            achievement = ACHIEVEMENTS[achievement_id]
            new_achievements.append({
                "id": achievement_id,
                "name": achievement["name"],
                "desc": achievement["desc"],
                "icon": achievement["icon"],
                "xp": achievement["xp"],
                "earned_at": datetime.utcnow().isoformat()
            })
            current_achievements.append(achievement_id)
    
    if new_achievements:
        total_bonus_xp = sum(a["xp"] for a in new_achievements)
        await database.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {"achievements": current_achievements},
                "$inc": {"xp": total_bonus_xp}
            }
        )
    
    return new_achievements

@router.post("/xp/award")
async def award_xp(action: XPAction, user = Depends(get_current_user)):
    """Award XP for user actions."""
    database = await get_database()
    user_id = str(user["_id"])
    
    # Get XP amount
    xp_amount = XP_ACTIONS.get(action.action, 0)
    if xp_amount == 0:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    # Build update query
    update_query: dict[str, Any] = {"$inc": {"xp": xp_amount}}
    
    # Track specific actions
    if action.action == "analyze_code":
        update_query["$inc"]["total_analyses"] = 1
    elif action.action == "run_code":
        update_query["$inc"]["total_runs"] = 1
    elif action.action == "ask_ai":
        update_query["$inc"]["total_questions"] = 1
    
    # Track language usage
    if action.language:
        update_query["$addToSet"] = {"languages_used": action.language}
    
    # Update streak
    today = datetime.utcnow().date().isoformat()
    last_active = user.get("streak_updated", "")
    
    if last_active != today:
        yesterday = (datetime.utcnow() - timedelta(days=1)).date().isoformat()
        if last_active == yesterday:
            # Continue streak
            update_query["$inc"]["streak"] = 1
            streak_bonus = XP_ACTIONS["streak_bonus"] * (user.get("streak", 0) + 1)
            update_query["$inc"]["xp"] += streak_bonus
            # Set streak update fields
            set_fields: dict[str, Any] = {"streak_updated": today, "last_active": datetime.utcnow()}
            update_query["$set"] = set_fields
        else:
            # Reset streak
            update_query["$set"] = {"streak": 1, "streak_updated": today, "last_active": datetime.utcnow()}
    
    # Apply update
    await database.users.update_one({"_id": ObjectId(user_id)}, update_query)
    
    # Get updated user data
    updated_user = await database.users.find_one({"_id": ObjectId(user_id)})
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate new level
    new_xp = updated_user.get("xp", 0)
    new_level = calculate_level(new_xp)
    old_level = user.get("level", 1)
    
    level_up = new_level > old_level
    if level_up:
        await database.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"level": new_level}}
        )
        updated_user["level"] = new_level
    
    # Check for new achievements
    new_achievements = await check_and_award_achievements(database, user_id, updated_user)
    
    return {
        "xp_earned": xp_amount,
        "total_xp": new_xp,
        "level": new_level,
        "level_up": level_up,
        "new_achievements": new_achievements,
        "streak": updated_user.get("streak", 0)
    }

@router.get("/stats", response_model=StatsResponse)
async def get_stats(user = Depends(get_current_user)):
    """Get user gamification stats."""
    database = await get_database()
    
    xp = user.get("xp", 0)
    level = calculate_level(xp)
    xp_needed = xp_to_next_level(xp)
    progress = ((xp % XP_PER_LEVEL) / XP_PER_LEVEL) * 100
    
    # Get user rank
    higher_xp_count = await database.users.count_documents({"xp": {"$gt": xp}})
    rank = higher_xp_count + 1
    
    # Get full achievement data
    earned_ids = user.get("achievements", [])
    achievements = []
    for aid in earned_ids:
        if aid in ACHIEVEMENTS:
            achievements.append({
                "id": aid,
                **ACHIEVEMENTS[aid]
            })
    
    return StatsResponse(
        xp=xp,
        level=level,
        xp_to_next_level=xp_needed,
        xp_progress_percent=int(progress),
        streak=user.get("streak", 0),
        total_analyses=user.get("total_analyses", 0),
        total_runs=user.get("total_runs", 0),
        total_questions=user.get("total_questions", 0),
        achievements=achievements,
        languages_used=user.get("languages_used", []),
        rank=rank
    )

@router.get("/achievements/all")
async def get_all_achievements():
    """Get all available achievements."""
    return [{"id": k, **v} for k, v in ACHIEVEMENTS.items()]

@router.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get top users by XP."""
    database = await get_database()
    
    cursor = database.users.find(
        {},
        {"name": 1, "xp": 1, "level": 1, "streak": 1, "achievements": 1}
    ).sort("xp", -1).limit(limit)
    
    leaderboard = []
    rank = 1
    async for user in cursor:
        leaderboard.append({
            "rank": rank,
            "name": user["name"],
            "xp": user.get("xp", 0),
            "level": calculate_level(user.get("xp", 0)),
            "streak": user.get("streak", 0),
            "achievements_count": len(user.get("achievements", []))
        })
        rank += 1
    
    return leaderboard
