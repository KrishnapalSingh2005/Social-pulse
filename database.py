import sqlite3
import json
import pandas as pd
from datetime import datetime, timedelta
import random
from auth import hash_password

DB_NAME = "socialverse_platform.db"

def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    """
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the enterprise-grade 5-table relational schema.
    Applies indexes for query optimization, and seeds default records if empty.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable foreign keys support in SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Users Table (Core profile registry)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        password_hash TEXT NOT NULL,
        bio TEXT,
        followers_count INTEGER DEFAULT 0,
        following_count INTEGER DEFAULT 0,
        avatar_url TEXT,
        is_verified INTEGER DEFAULT 0,
        is_admin INTEGER DEFAULT 0
    )
    """)

    # 2. Posts Table (Content metadata ledger)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        post_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        content TEXT NOT NULL,
        sentiment_score REAL DEFAULT 0.0,
        is_toxic INTEGER DEFAULT 0,
        posted_date TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    )
    """)

    # 3. Engagement Metrics Table (Separates transactional metrics from content)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS engagement_metrics (
        metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER UNIQUE NOT NULL,
        likes INTEGER DEFAULT 0,
        comments INTEGER DEFAULT 0,
        shares INTEGER DEFAULT 0,
        views INTEGER DEFAULT 0,
        clicks INTEGER DEFAULT 0,
        engagement_rate REAL DEFAULT 0.0,
        is_fake_engagement INTEGER DEFAULT 0,
        FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
    )
    """)

    # 4. AI Predictions Log Table (Audit trail of ML pipelines)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ai_predictions_log (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_features TEXT NOT NULL,
        prediction_class TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)

    # 5. Activity Logs Table (Security and telemetry logging)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activity_logs (
        activity_id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        user TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """)

    # 6. Moderation Logs Table (Admin review pipeline)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS moderation (
        mod_id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        reason TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        timestamp TEXT NOT NULL,
        FOREIGN KEY (post_id) REFERENCES posts(post_id) ON DELETE CASCADE
    )
    """)

    # Create Indexes for Query Optimization
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_user_id ON posts(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(posted_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_post_id ON engagement_metrics(post_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activities_type ON activity_logs(type);")

    conn.commit()

    # Check and Seed Datasets
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        print("Database is empty. Initializing advanced seeds...")
        seed_database(conn)

    conn.close()

def seed_database(conn):
    """
    Seeds database with mock profiles, detailed transactional metrics,
    historical prediction logs, and audit logs.
    """
    cursor = conn.cursor()

    # Pre-hash secure demonstration credentials
    admin_pw = hash_password("admin123")
    user_pw = hash_password("user123")

    # Seed verified and unverified profiles
    users_data = [
        # username, full_name, password_hash, bio, followers, following, avatar_url, is_verified, is_admin
        ("admin", "System Administrator", admin_pw, "Core operations supervisor, security gate, and data engineer.", 54100, 110, "https://api.dicebear.com/7.x/bottts/svg?seed=admin", 1, 1),
        ("alex_codes", "Alex Mercer", user_pw, "Lead Full-Stack Developer. Writing optimized database schemas and CSS.", 15200, 480, "https://api.dicebear.com/7.x/bottts/svg?seed=alex", 1, 0),
        ("sarah_sec", "Sarah Jenkins", user_pw, "Cybersecurity Architect. Specializing in cloud network vulnerability scanning.", 8400, 210, "https://api.dicebear.com/7.x/bottts/svg?seed=sarah", 1, 0),
        ("data_dan", "Dan Patel", user_pw, "Senior Data Scientist. Building regression pipelines and training trees.", 24100, 950, "https://api.dicebear.com/7.x/bottts/svg?seed=dan", 1, 0),
        ("lisa_ux", "Lisa Chen", user_pw, "Creative Interface Designer. Obsessed with high-fidelity glassmorphic visuals.", 18900, 1100, "https://api.dicebear.com/7.x/bottts/svg?seed=lisa", 1, 0),
        ("tech_traveler", "Marcus Sterling", user_pw, "Digital Nomad and Consultant. Writing python scripts in beachside cafes. ☕✈️", 9600, 720, "https://api.dicebear.com/7.x/bottts/svg?seed=marcus", 0, 0)
    ]

    cursor.executemany("""
    INSERT INTO users (username, full_name, password_hash, bio, followers_count, following_count, avatar_url, is_verified, is_admin)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, users_data)
    conn.commit()

    # Map user IDs
    cursor.execute("SELECT user_id, username FROM users")
    user_map = {row["username"]: row["user_id"] for row in cursor.fetchall()}

    # Diverse posts with pre-computed Sentiment & Toxicity profiles
    posts_data = [
        # alex_codes
        ("alex_codes", "FastAPI is incredibly elegant. Creating REST APIs with relational SQLite models and serving glowing frontend dashboards takes college assignments to the next level!", 0.85, 0, 1),
        ("alex_codes", "Debugging circular import imports in python at 3 AM. It feels like chasing a ghost in a machine. 😭💻 #developer", -0.4, 0, 3),
        ("alex_codes", "Always keep your functions focused, short, and modular. Writing clean standard code is a superpower in startups.", 0.6, 0, 5),
        ("alex_codes", "Your code is completely stupid, brainless, and toxic trash. Please delete your repository and quit programming.", -0.9, 1, 8), # Toxic!

        # sarah_sec
        ("sarah_sec", "Enterprise PSA: Audit your network perimeters immediately. A new exploit chain bypasses default firewall routing tables.", 0.3, 0, 2),
        ("sarah_sec", "Investigating a sophisticated memory injection attack today. Secure token validation is not optional.", 0.1, 0, 6),
        ("sarah_sec", "Debugging firewall ports only to find a typo in our cloud server configurations. Typical hacker day.", -0.2, 0, 10),

        # data_dan
        ("data_dan", "Stunning breakthrough in large language model weights compression. Running 70B parameter models on single GPU nodes is now reality! 🚀", 0.9, 0, 2),
        ("data_dan", "Remember: Tuning hyper-parameters is useless if your training database has dirty data bias. Invest in cleaning your databases first!", 0.5, 0, 4),
        ("data_dan", "A visual explanation of Bias-Variance Tradeoff in Deep Neural Networks. Slide left to view the code! ➡️", 0.4, 0, 7),
        ("data_dan", "Warning: High comment-to-like ratios detected. An anomalous botnet spam attempt is likely occurring.", -0.3, 0, 9), # Anomalous engagement!

        # lisa_ux
        ("lisa_ux", "Future UI Style: Dark space backgrounds paired with vibrant cyber cyan glows and soft glassmorphism borders. Linear dashboard vibes.", 0.95, 0, 1),
        ("lisa_ux", "Great user experience is invisible; user navigation flows should feel natural and friction-free.", 0.75, 0, 4),
        ("lisa_ux", "Working on complex responsive glassmorphism CSS templates is therapeutic. The depth and shadows look incredibly premium.", 0.8, 0, 8),

        # tech_traveler
        ("tech_traveler", "Waking up to this sunrise view in Bali. Designing SaaS databases from a laptop on the beach makes nomad consulting so rewarding. 🌊🌴", 0.98, 0, 3),
        ("tech_traveler", "Nomad Gear: Ultra-slim travel adapter, noise-cancelling headphones, and a reliable e-SIM keep operations scaling smoothly.", 0.5, 0, 6),
        ("tech_traveler", "How to manage overseas remote clients across conflicting timezones without burning out. New article dropping soon!", 0.2, 0, 9)
    ]

    # Pre-calculated engagement stats for sub-table
    engagement_stats = {
        # posts content indices mapping
        1: (850, 190, 112, 1200, 350, 0),  # Likes, Comments, Shares, Views, Clicks, is_fake
        2: (120, 35, 12, 340, 45, 0),
        3: (420, 58, 22, 980, 140, 0),
        4: (35, 120, 2, 450, 8, 0), # Low likes, high comments (toxic post)
        5: (950, 210, 340, 2500, 850, 0),
        6: (280, 42, 18, 720, 95, 0),
        7: (190, 15, 3, 410, 32, 0),
        8: (1420, 450, 510, 4200, 1100, 0),
        9: (880, 140, 95, 2100, 450, 0),
        10: (1100, 190, 88, 3100, 620, 0),
        11: (45, 480, 3, 1400, 20, 1), # High comments, low likes (Anomalous fake engagement!)
        12: (1680, 320, 450, 4500, 1420, 0),
        13: (940, 110, 88, 2800, 520, 0),
        14: (620, 68, 38, 1900, 280, 0),
        15: (2100, 480, 190, 5600, 1850, 0),
        16: (480, 38, 12, 1100, 110, 0),
        17: (380, 45, 18, 920, 98, 0)
    }

    # Insert posts and metrics
    for idx, (username, content, sentiment, toxic, days_ago) in enumerate(posts_data, 1):
        user_id = user_map[username]
        # Generate dynamic dates
        posted_date = (datetime.now() - timedelta(days=days_ago, hours=random.randint(0,23))).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO posts (post_id, user_id, content, sentiment_score, is_toxic, posted_date)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (idx, user_id, content, sentiment, toxic, posted_date))

        # Get metrics
        likes, comments, shares, views, clicks, is_fake = engagement_stats.get(idx, (100, 10, 5, 300, 20, 0))
        # Rate: (likes + comments + shares) / views * 100
        rate = round(((likes + comments + shares) / max(1, views)) * 100, 2)

        cursor.execute("""
        INSERT INTO engagement_metrics (post_id, likes, comments, shares, views, clicks, engagement_rate, is_fake_engagement)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (idx, likes, comments, shares, views, clicks, rate, is_fake))

    # Seed Moderation log
    cursor.execute("""
    INSERT INTO moderation (post_id, reason, status, timestamp)
    VALUES (4, 'Flagged automatically by AI Toxicity scanner: abusive profanity.', 'flagged', ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),))

    # Seed Activities Logs
    activities = [
        ("system", "SYS", "SocialVerse Core server initialized with 5-table relational SQLite layout.", (datetime.now() - timedelta(minutes=45)).strftime("%H:%M:%S")),
        ("security", "SYS", "Bcrypt password hashing context loaded securely.", (datetime.now() - timedelta(minutes=35)).strftime("%H:%M:%S")),
        ("auth", "admin", "Admin session established with secure JWT credentials.", (datetime.now() - timedelta(minutes=25)).strftime("%H:%M:%S")),
        ("fraud", "data_dan", "Platform anomaly scanner flagged Post #11 for high comment-to-like ratio.", (datetime.now() - timedelta(minutes=15)).strftime("%H:%M:%S")),
        ("mod", "AI_SCAN", "Post #4 flagged for abusive language and routed to moderation console.", (datetime.now() - timedelta(minutes=5)).strftime("%H:%M:%S"))
    ]
    cursor.executemany("INSERT INTO activity_logs (type, user, content, timestamp) VALUES (?, ?, ?, ?)", activities)

    # Seed AI Predictions logs
    predictions = [
        ('{"likes":120, "comments":18, "shares":5}', "Medium", 91.5, (datetime.now() - timedelta(minutes=20)).strftime("%H:%M:%S")),
        ('{"likes":2100, "comments":480, "shares":190}', "High", 98.4, (datetime.now() - timedelta(minutes=10)).strftime("%H:%M:%S"))
    ]
    cursor.executemany("INSERT INTO ai_predictions_log (input_features, prediction_class, confidence_score, timestamp) VALUES (?, ?, ?, ?)", predictions)

    conn.commit()

# --- Advanced Analytical SQL Aggregations & CTE Pipelines ---

def db_fetch_metrics():
    """
    Advanced query aggregate. Extracts total users, posts, cumulative metrics,
    activity hours, and computes dynamic platform metrics.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Dynamic DAU (Daily Active Users): Unique posters + active users in logs
    cursor.execute("""
    SELECT COUNT(DISTINCT user_id) FROM posts 
    WHERE posted_date >= date('now', '-7 days')
    """)
    dau = cursor.fetchone()[0] or 1
    # Adding a small randomized base to represent platform background actions
    dau = min(6, dau + 2)

    # Core aggregations joining posts & metrics tables
    cursor.execute("""
    SELECT 
        COUNT(DISTINCT p.user_id) as total_users,
        COUNT(p.post_id) as total_posts,
        SUM(m.likes) as total_likes,
        SUM(m.comments) as total_comments,
        SUM(m.shares) as total_shares,
        SUM(m.views) as total_views,
        AVG(m.engagement_rate) as avg_rate
    FROM posts p
    LEFT JOIN engagement_metrics m ON p.post_id = m.post_id
    """)
    row = cursor.fetchone()
    
    # Peak engagement hour (SQL strftime extractor + SUM order limit)
    cursor.execute("""
    SELECT strftime('%H', p.posted_date) as post_hour, SUM(m.likes + m.comments + m.shares) as total_eng
    FROM posts p
    JOIN engagement_metrics m ON p.post_id = m.post_id
    GROUP BY post_hour
    ORDER BY total_eng DESC
    LIMIT 1
    """)
    peak_row = cursor.fetchone()
    peak_hour = f"{peak_row['post_hour']}:00 EST" if peak_row else "18:00 EST"

    # Virality Factor (proportions of high virality posts)
    cursor.execute("SELECT COUNT(*) FROM engagement_metrics WHERE likes > 1000")
    viral_count = cursor.fetchone()[0]
    viral_score = min(98.5, round(12.4 + (viral_count * 14.8), 1))

    conn.close()

    return {
        "users": row["total_users"] or 0,
        "posts": row["total_posts"] or 0,
        "likes": row["total_likes"] or 0,
        "comments": row["total_comments"] or 0,
        "shares": row["total_shares"] or 0,
        "views": row["total_views"] or 0,
        "engagement_rate": round(row["avg_rate"] or 0.0, 2),
        "dau": dau,
        "peak_hour": peak_hour,
        "viral_score": viral_score
    }

def db_fetch_users_with_intelligence():
    """
    Advanced SQL with Common Table Expressions (CTEs).
    Calculates dynamic User Influence Index and Community Impact Score.
    
    Formula: Influence = Followers * 0.45 + Likes * 0.35 + AvgRate * 0.2
    """
    conn = get_db_connection()
    query = """
    WITH UserAggregates AS (
        SELECT 
            p.user_id,
            COUNT(p.post_id) as total_posts,
            SUM(m.likes) as total_likes,
            SUM(m.comments) as total_comments,
            SUM(m.shares) as total_shares,
            AVG(m.engagement_rate) as avg_rate
        FROM posts p
        LEFT JOIN engagement_metrics m ON p.post_id = m.post_id
        GROUP BY p.user_id
    )
    SELECT 
        u.user_id, u.username, u.full_name, u.bio, u.avatar_url,
        u.followers_count, u.following_count, u.is_verified, u.is_admin,
        COALESCE(ua.total_posts, 0) as total_posts,
        COALESCE(ua.total_likes, 0) as total_likes,
        COALESCE(ua.total_comments, 0) as total_comments,
        COALESCE(ua.total_shares, 0) as total_shares,
        ROUND(COALESCE(ua.avg_rate, 0.0), 2) as avg_engagement_rate,
        -- Calculate Influence Index
        ROUND(
            (u.followers_count * 0.001 * 45) + 
            (COALESCE(ua.total_likes, 0) * 0.01 * 35) + 
            (COALESCE(ua.avg_rate, 0.0) * 20)
        , 1) as influence_index
    FROM users u
    LEFT JOIN UserAggregates ua ON u.user_id = ua.user_id
    ORDER BY influence_index DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate community impact score: influence rank plus small activity bonus
    df["community_impact"] = df["influence_index"].apply(lambda x: min(99.0, round(x * 1.15 + 5.0, 1)))
    return df.to_dict(orient="records")

def db_fetch_posts_analytics_feed():
    """
    Advanced JOIN query merging posts, users, and detailed engagement metrics.
    """
    conn = get_db_connection()
    query = """
    SELECT 
        p.post_id, p.content, p.sentiment_score, p.is_toxic, p.posted_date,
        u.username, u.full_name, u.avatar_url, u.is_verified,
        m.likes, m.comments, m.shares, m.views, m.clicks, m.engagement_rate, m.is_fake_engagement
    FROM posts p
    JOIN users u ON p.user_id = u.user_id
    JOIN engagement_metrics m ON p.post_id = m.post_id
    ORDER BY p.posted_date DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")

def db_fetch_correlation_matrix():
    """
    Calculates analytical Pearson correlations coefficients between
    likes, comments, shares, views, and clicks using Pandas.
    Demonstrates analytical data intelligence.
    """
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT likes, comments, shares, views, clicks FROM engagement_metrics", conn)
    conn.close()

    if df.empty:
        return []

    # Calculate Pearson correlations
    corr = df.corr().round(2).fillna(0.0)
    
    # Format for visual frontend matrix
    matrix = []
    features = ["likes", "comments", "shares", "views", "clicks"]
    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            matrix.append({
                "x": f1.upper(),
                "y": f2.upper(),
                "val": float(corr.loc[f1, f2])
            })
    return matrix

def db_fetch_peak_engagement_hours():
    """
    SQL strftime query extracts publication hours, aggregates cumulative
    engagements, and maps traffic distribution.
    """
    conn = get_db_connection()
    query = """
    SELECT 
        strftime('%H', p.posted_date) as hour,
        SUM(m.likes + m.comments + m.shares) as total_engagement,
        AVG(m.engagement_rate) as avg_rate
    FROM posts p
    JOIN engagement_metrics m ON p.post_id = m.post_id
    GROUP BY hour
    ORDER BY hour ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")

def db_log_prediction(features_dict, prediction_class, confidence):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%H:%M:%S")
    features_json = json.dumps(features_dict)
    
    cursor.execute("""
    INSERT INTO ai_predictions_log (input_features, prediction_class, confidence_score, timestamp)
    VALUES (?, ?, ?, ?)
    """, (features_json, prediction_class, confidence, timestamp))
    conn.commit()
    conn.close()

def db_fetch_predictions_log(limit=25):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT prediction_id, input_features, prediction_class, confidence_score, timestamp FROM ai_predictions_log ORDER BY prediction_id DESC LIMIT {limit}")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def db_fetch_activities_log(limit=15):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT type, user, content, timestamp FROM activity_logs ORDER BY activity_id DESC LIMIT {limit}")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def db_log_activity(act_type, user, content):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%H:%M:%S")
    cursor.execute("""
    INSERT INTO activity_logs (type, user, content, timestamp)
    VALUES (?, ?, ?, ?)
    """, (act_type, user, content, timestamp))
    conn.commit()
    conn.close()

def db_fetch_moderation():
    conn = get_db_connection()
    query = """
    SELECT 
        m.mod_id, m.reason, m.status, m.timestamp,
        p.post_id, p.content, u.username
    FROM moderation m
    JOIN posts p ON m.post_id = p.post_id
    JOIN users u ON p.user_id = u.user_id
    ORDER BY m.mod_id DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df.to_dict(orient="records")

def db_update_moderation(mod_id, status):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE moderation SET status = ? WHERE mod_id = ?", (status, mod_id))
    
    if status == "approved":
        cursor.execute("SELECT post_id FROM moderation WHERE mod_id = ?", (mod_id,))
        post_id = cursor.fetchone()[0]
        cursor.execute("UPDATE posts SET is_toxic = 0 WHERE post_id = ?", (post_id,))
        
    conn.commit()
    conn.close()
    return True

def db_create_post(username, content, likes=0, comments=0, shares=0, sentiment=0.0, toxic=0, fake=0):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Find user ID
    cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None
    user_id = row["user_id"]
    posted_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Insert into core posts table
    cursor.execute("""
    INSERT INTO posts (user_id, content, sentiment_score, is_toxic, posted_date)
    VALUES (?, ?, ?, ?, ?)
    """, (user_id, content, sentiment, toxic, posted_date))
    new_post_id = cursor.lastrowid
    
    # 2. Insert into metrics table
    views = max(100, (likes + comments + shares) * random.randint(2, 5))
    clicks = int(views * random.uniform(0.05, 0.25))
    rate = round(((likes + comments + shares) / max(1, views)) * 100, 2)
    
    cursor.execute("""
    INSERT INTO engagement_metrics (post_id, likes, comments, shares, views, clicks, engagement_rate, is_fake_engagement)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (new_post_id, likes, comments, shares, views, clicks, rate, fake))
    
    conn.commit()
    
    # If toxic, log in moderation
    if toxic == 1:
        cursor.execute("""
        INSERT INTO moderation (post_id, reason, status, timestamp)
        VALUES (?, ?, 'flagged', ?)
        """, (new_post_id, "Flagged automatically by AI Toxicity scanner: profanity hits", posted_date))
        conn.commit()
        
    conn.close()
    return new_post_id

def db_create_user(username, full_name, password, bio=""):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed = hash_password(password)
    avatar = f"https://api.dicebear.com/7.x/bottts/svg?seed={username}"
    
    try:
        cursor.execute("""
        INSERT INTO users (username, full_name, password_hash, bio, followers_count, following_count, avatar_url, is_verified, is_admin)
        VALUES (?, ?, ?, ?, 1200, 310, ?, 0, 0)
        """, (username, full_name, hashed, bio, avatar))
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return new_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def db_validate_login(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_id, username, full_name, password_hash, is_admin FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
        
    from auth import verify_password
    if verify_password(password, row["password_hash"]):
        return {
            "user_id": row["user_id"],
            "username": row["username"],
            "full_name": row["full_name"],
            "is_admin": row["is_admin"]
        }
    return None
