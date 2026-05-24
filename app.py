import io
import time
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd

# Core Project Modules
from database import (
    init_db,
    get_db_connection,
    db_fetch_metrics,
    db_fetch_users_with_intelligence,
    db_fetch_posts_analytics_feed,
    db_fetch_correlation_matrix,
    db_fetch_predictions_log,
    db_fetch_activities_log,
    db_log_activity,
    db_fetch_moderation,
    db_update_moderation,
    db_create_post,
    db_create_user,
    db_validate_login,
)
from ai_engine import (
    analyze_sentiment,
    scan_toxicity,
    train_popularity_classifier,
    forecast_user_growth,
    detect_fake_engagement,
    generate_user_recommendations,
    generate_ai_caption,
)
from auth import (
    create_access_token,
    decode_access_token,
    get_current_user,
    get_admin_user,
)

# Initialize FastAPI App
app = FastAPI(
    title="SocialVerse Data Intelligence & AI Platform API",
    description="Enterprise-grade analytics, scikit-learn ML models, and secure JWT-based profiles ledger.",
    version="1.0.0"
)

# Enable CORS for robust cross-origin testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database schema and seeds on startup
@app.on_event("startup")
def startup_db_check():
    print("[SYSTEM] Initializing secure 5-table relational SQLite layout...")
    init_db()
    print("[SYSTEM] Relational database schemas initialized successfully.")

# ----------------------------------------------------
# PYDANTIC SCHEMAS / REQUEST DRAFT SCHEMAS
# ----------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    full_name: str
    password: str
    bio: Optional[str] = ""

class PostCreateRequest(BaseModel):
    content: str
    likes: int
    comments: int
    shares: int

class PredictionRequest(BaseModel):
    likes: int
    comments: int
    shares: int

class CaptionRequest(BaseModel):
    niche: str

class RecommendRequest(BaseModel):
    bio: str

class ModerationActionRequest(BaseModel):
    mod_id: int
    status: str

# ----------------------------------------------------
# 1. AUTHENTICATION & SECURITY SERVICES
# ----------------------------------------------------
@app.post("/api/auth/login")
def auth_login(payload: LoginRequest):
    user = db_validate_login(payload.username, payload.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session authorization failed. Invalid security key or username."
        )
    
    # Generate secure JWT session token
    token = create_access_token({
        "user_id": user["user_id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "is_admin": bool(user["is_admin"])
    })
    
    db_log_activity("auth", user["username"], "JWT authorized session established.")
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "is_admin": bool(user["is_admin"])
        }
    }

@app.post("/api/auth/register")
def auth_register(payload: RegisterRequest):
    new_user_id = db_create_user(
        username=payload.username,
        full_name=payload.full_name,
        password=payload.password,
        bio=payload.bio
    )
    if not new_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Node registration failed. Username is already indexed."
        )
    
    db_log_activity("auth", payload.username, "New user node registered in database ledger.")
    return {"success": True, "message": "Creator profile node successfully written."}

# ----------------------------------------------------
# 2. SOCIAL ANALYTICS PIPELINES
# ----------------------------------------------------
@app.get("/api/analytics/overview")
def get_analytics_overview():
    try:
        metrics = db_fetch_metrics()
        matrix = db_fetch_correlation_matrix()
        return {
            "metrics": metrics,
            "correlation_matrix": matrix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch SaaS dashboard aggregates: {str(e)}")

@app.get("/api/analytics/users")
def get_analytics_users():
    try:
        return db_fetch_users_with_intelligence()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch CTE rankings: {str(e)}")

@app.get("/api/analytics/user/{user_id}")
def get_user_intelligence(user_id: int):
    users = db_fetch_users_with_intelligence()
    user = next((u for u in users if u["user_id"] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="Creator node not found inside platform indexes")
    
    followers = user.get("followers_count", 1200)
    posts = user.get("total_posts", 5)
    
    # Run ML Followers Linear Regression Forecaster Curve
    report = forecast_user_growth(followers, posts)
    
    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "forecaster_report": report
    }

# ----------------------------------------------------
# 3. POSTS FEED TRANSACTION SERVICES
# ----------------------------------------------------
@app.get("/api/posts/list")
def get_posts_list():
    try:
        return db_fetch_posts_analytics_feed()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch joined platform feed: {str(e)}")

@app.post("/api/posts/create")
def create_post(payload: PostCreateRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    content = payload.content
    likes = payload.likes
    comments = payload.comments
    shares = payload.shares
    
    # 1. Run real-time AI lexical content audits
    sentiment = analyze_sentiment(content)
    is_toxic, confidence = scan_toxicity(content)
    
    # 2. Check for suspicious simulated fake activities
    is_fake = detect_fake_engagement(likes, comments, shares)
    
    # 3. Write record into database ledger
    new_post_id = db_create_post(
        username=username,
        content=content,
        likes=likes,
        comments=comments,
        shares=shares,
        sentiment=sentiment,
        toxic=is_toxic,
        fake=is_fake
    )
    if not new_post_id:
        raise HTTPException(status_code=500, detail="Database write error. Post record failed to compile.")
    
    # Log details into Telemetry system logs
    db_log_activity("post", username, f"Indexed post #{new_post_id}: '{content[:25]}...'")
    
    if is_toxic == 1:
        db_log_activity("mod", "AI_SCAN", f"Post #{new_post_id} toxicity audit hit critical threshold. Routed to admin queue.")
        
    if is_fake == 1:
        db_log_activity("fraud", "AI_SCAN", f"Post #{new_post_id} flagged for botnet comment spam spike anomaly.")
        
    return {"success": True, "post_id": new_post_id}

# ----------------------------------------------------
# 4. AI & SCIKIT-LEARN ML SANDBOX ENDPOINTS
# ----------------------------------------------------
@app.post("/api/ai/predict")
def run_ai_prediction(payload: PredictionRequest):
    try:
        # Load feed history
        posts = db_fetch_posts_analytics_feed()
        
        # Train classifier dynamically
        model, weights, accuracy = train_popularity_classifier(posts)
        
        # Prepare inputs dataframe for scikit-learn DT model
        X_predict = pd.DataFrame(
            [[payload.likes, payload.comments, payload.shares]], 
            columns=["likes", "comments", "shares"]
        )
        prediction_class = model.predict(X_predict)[0]
        
        # Log prediction to DB audit
        db_log_prediction(
            features_dict={"likes": payload.likes, "comments": payload.comments, "shares": payload.shares},
            prediction_class=prediction_class,
            confidence=accuracy * 100
        )
        
        # Total engagement score
        engagement_score = payload.likes * 1 + payload.comments * 2 + payload.shares * 3
        
        return {
            "prediction": prediction_class.upper(),
            "engagement_score": engagement_score,
            "confidence_pct": int(accuracy * 100)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML Popularity Inference pipeline crash: {str(e)}")

@app.post("/api/ai/generate-caption")
def get_ai_caption(payload: CaptionRequest):
    caption = generate_ai_caption(payload.niche)
    return {"caption": caption}

@app.post("/api/ai/recommend")
def get_ai_recommendations(payload: RecommendRequest):
    recs = generate_user_recommendations(payload.bio)
    return {"recommendations": recs}

# ----------------------------------------------------
# 5. ADMINISTRATIVE CONTROL OPERATIONS
# ----------------------------------------------------
@app.get("/api/admin/system/health")
def get_system_health(current_user: dict = Depends(get_admin_user)):
    start_time = time.time()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    cursor.fetchone()
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    cursor.execute("SELECT COUNT(*) FROM users")
    users_records = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM posts")
    posts_records = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "latency_ms": latency_ms,
        "database": "SQLite3 (socialverse_platform.db)",
        "table_stats": {
            "users_records": users_records,
            "posts_records": posts_records
        }
    }

@app.get("/api/admin/moderation/list")
def get_moderation_queue(current_user: dict = Depends(get_admin_user)):
    try:
        return db_fetch_moderation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch moderation queue: {str(e)}")

@app.post("/api/admin/moderation/action")
def execute_moderation_action(payload: ModerationActionRequest, current_user: dict = Depends(get_admin_user)):
    try:
        success = db_update_moderation(payload.mod_id, payload.status)
        db_log_activity("mod", current_user["username"], f"Moderation ledger ID #{payload.mod_id} status updated to {payload.status}.")
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation queue action failed: {str(e)}")

@app.get("/api/admin/predictions-log")
def get_predictions_audit_log(current_user: dict = Depends(get_admin_user)):
    try:
        return db_fetch_predictions_log(25)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytical prediction logs: {str(e)}")

@app.get("/api/admin/export/csv")
def stream_database_csv(token: str = Query(...)):
    # Validate token query directly (essential for direct window.open link streams)
    payload = decode_access_token(token)
    if not payload or not payload.get("is_admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Clearance denied. Administrative session required to fetch telemetry."
        )
    
    try:
        conn = get_db_connection()
        query = """
        SELECT 
            p.post_id, p.content, p.sentiment_score, p.is_toxic, p.posted_date,
            u.username, u.full_name,
            m.likes, m.comments, m.shares, m.views, m.clicks, m.engagement_rate, m.is_fake_engagement
        FROM posts p
        JOIN users u ON p.user_id = u.user_id
        JOIN engagement_metrics m ON p.post_id = m.post_id
        ORDER BY p.post_id ASC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Stream dataframe as in-memory CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        return StreamingResponse(
            iter([csv_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=socialverse_export_data.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compile database export stream: {str(e)}")

# ----------------------------------------------------
# 6. GLOBAL ACTIVITIES / TELEMETRY LOGS
# ----------------------------------------------------
@app.get("/api/activities/feed")
def get_activities_feed():
    try:
        return db_fetch_activities_log(15)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch system logs: {str(e)}")

# ----------------------------------------------------
# 7. ROUTING TO SINGLE-PAGE APPLICATION FRONTEND
# ----------------------------------------------------
@app.get("/")
def serve_spa():
    return FileResponse("static/index.html")

# Mount remaining static folders
app.mount("/static", StaticFiles(directory="static"), name="static")
