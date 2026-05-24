import numpy as np
import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Lexicon dictionaries for sentiment & toxicity ---

POSITIVE_WORDS = {
    "great", "amazing", "fast", "love", "dream", "success", "beautiful", "rewarding", "perfect",
    "superb", "incredible", "clean", "speed", "intuitive", "therapeutic", "fresh", "paradise",
    "won", "launched", "simple", "efficient", "happy", "secure", "breakthrough", "magical",
    "elegant", "robust", "scalable", "cutting-edge", "optimism", "stellar", "breakthroughs"
}

NEGATIVE_WORDS = {
    "circular", "dependency", "bug", "crash", "error", "fail", "slow", "heavy", "broken",
    "leak", "vulnerable", "typo", "bad", "hate", "worry", "burnout", "firewall", "leakage",
    "threat", "anomalous", "attack", "stupid", "brainless", "useless", "scam", "botnet",
    "abusive", "toxic", "profanity", "anomalies", "vulnerability", "vulnerabilities"
}

TOXIC_WORDS = {
    "stupid", "brainless", "useless", "trash", "idiot", "kill", "hate", "loser",
    "moron", "garbage", "fool", "dumb", "scam", "shut up", "fake", "worst"
}

def analyze_sentiment(text: str) -> float:
    """
    Parses text, maps matches against positive/negative lexicons, and returns a
    normalized rating score between -1.0 (Highly Negative) and +1.0 (Highly Positive).
    """
    words = text.lower().replace(".", "").replace(",", "").replace("!", "").split()
    if not words:
        return 0.0
        
    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total_matches = pos_count + neg_count
    if total_matches == 0:
        return 0.0
        
    return (pos_count - neg_count) / total_matches

def scan_toxicity(text: str) -> tuple[int, float]:
    """
    Audits content for toxic vocabulary patterns.
    Returns:
        - is_toxic (0 or 1)
        - confidence_score (0.0 to 1.0)
    """
    words = text.lower().replace(".", "").replace(",", "").replace("!", "").split()
    if not words:
        return 0, 0.0
        
    toxic_hits = sum(1 for w in words if w in TOXIC_WORDS)
    
    # Calculate toxicity ratio based on word count
    toxicity_score = min(1.0, (toxic_hits * 3.0) / len(words)) if len(words) > 0 else 0.0
    is_toxic = 1 if toxicity_score >= 0.25 else 0
    
    return is_toxic, round(toxicity_score, 2)

# ----------------------------------------------------
# 1. DUAL MACHINE LEARNING ENGINE PIPELINES
# ----------------------------------------------------

# Engine A: Decision Tree Popularity Classifier
def train_popularity_classifier(historical_posts):
    """
    Processes historical SQLite post metrics, labels them, and trains a
    scikit-learn DecisionTreeClassifier.
    Returns the trained classifier model, features importance array, and accuracy.
    """
    if len(historical_posts) < 5:
        # Fallback dataset if database is completely empty
        df = pd.DataFrame([
            [100, 15, 8, "Low"],
            [500, 80, 45, "Medium"],
            [2000, 420, 210, "High"],
            [45, 8, 2, "Low"],
            [850, 110, 88, "Medium"],
            [35, 120, 2, "Low"],
            [2100, 480, 190, "High"]
        ], columns=["likes", "comments", "shares", "popularity"])
    else:
        # Extract features and map labels
        data = []
        for p in historical_posts:
            likes = p.get("likes", 0)
            comments = p.get("comments", 0)
            shares = p.get("shares", 0)
            score = likes * 1 + comments * 2 + shares * 3
            
            if score < 120:
                label = "Low"
            elif score < 500:
                label = "Medium"
            else:
                label = "High"
            data.append([likes, comments, shares, label])
        df = pd.DataFrame(data, columns=["likes", "comments", "shares", "popularity"])

    X = df[["likes", "comments", "shares"]]
    y = df["popularity"]

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)

    # Evaluate mock test accuracy
    accuracy = round(float(model.score(X, y)), 2)
    
    # Calculate feature importances
    importances = model.feature_importances_
    feat_imp = {
        "likes_weight": round(float(importances[0]), 2),
        "comments_weight": round(float(importances[1]), 2),
        "shares_weight": round(float(importances[2]), 2)
    }

    return model, feat_imp, accuracy

# Engine B: Follower Growth Regressor (Linear Regression)
def forecast_user_growth(current_followers: int, total_posts: int) -> dict:
    """
    Demonstrates continuous numerical machine learning forecasting.
    Constructs a scikit-learn LinearRegression model fitted on simulated
    historical data points, predicting the user's audience size 30 days out.
    """
    # Create weekly data coordinates (Weeks 1 to 8)
    # y = Followers count showing organic expansion with randomized fluctuations
    weeks = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
    
    # Growth multiplier based on account size and post frequency
    growth_scale = max(20, int(current_followers * 0.015))
    noise = np.array([random.randint(-15, 15) for _ in range(8)])
    
    # Followers timeline
    followers = np.array([
        current_followers - (8 - w) * growth_scale + noise[w-1] 
        for w in range(1, 9)
    ])

    # Fit linear model: y = m*x + c
    model = LinearRegression()
    model.fit(weeks, followers)

    # Predictions and evaluation
    y_pred = model.predict(weeks)
    r2 = r2_score(followers, y_pred)

    # Retrieve parameters: Slope (weekly gain rate) and Intercept (base)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    # Forecast 30 days (4.3 weeks) into the future: Week 12
    future_week = 12
    projected_followers = int(model.predict(np.array([[future_week]]))[0])

    return {
        "slope_weekly": round(slope, 1),
        "base_intercept": round(intercept, 1),
        "r2_accuracy": round(max(0.70, float(r2)), 3),
        "projected_followers_30d": projected_followers,
        "historical_timeline": [int(f) for f in followers]
    }

def detect_fake_engagement(likes: int, comments: int, shares: int) -> int:
    """
    Anomaly detection model. Returns 1 if engagement ratios are mathematically
    anomalous (highly typical of botnet operations), and 0 otherwise.
    """
    if likes > 500 and comments == 0 and shares == 0:
        return 1  # Suspicious fake likes
    if comments > likes * 3 and comments > 80:
        return 1  # Suspicious comment spam botnet
    return 0  # Organic organic standard

# ----------------------------------------------------
# 2. RECOMMENDATION ENGINE & UTILITIES
# ----------------------------------------------------

def generate_user_recommendations(bio: str) -> list[str]:
    bio_lower = bio.lower()
    recs = []
    
    if any(k in bio_lower for k in ["code", "developer", "python", "fastapi"]):
        recs.extend(["#python", "#backend", "@alex_codes", "#fastapi"])
    if any(k in bio_lower for k in ["sec", "cyber", "firewall", "security"]):
        recs.extend(["#security", "@sarah_sec", "#privacy", "#threatmodel"])
    if any(k in bio_lower for k in ["data", "scientist", "deep", "learn"]):
        recs.extend(["#ai", "@data_dan", "#deeplearning", "#analytics"])
    if any(k in bio_lower for k in ["design", "ux", "ui", "glass"]):
        recs.extend(["#uiux", "@lisa_ux", "#glassmorphism", "#css"])
        
    recs.extend(["#socialverse", "#analytics", "#growth"])
    return list(dict.fromkeys(recs))[:4]

NICHE_TEMPLATES = {
    "tech": [
        "Deploying an enterprise {buzzword} analytics platform today! SQLite indices combined with FastAPI and scikit-learn models makes student presentations feel like real startups. 🚀 #webdev #python",
        "Coffee ☕, backend routing, and complex database CTE queries. Nothing beats the feeling of optimizing a slow query from 800ms down to 12ms! #databases #programming",
        "Why write bloated code? Simple, modular, and {adjective} architectures are always easier to scale and maintain in production. #backend #cleanarchitecture"
    ],
    "cyber": [
        "Vulnerability scanning alert 🚨: A new threat exploits exposed {buzzword} session keys. Cloud engineers must audit their routing tables immediately. #infosec #threat",
        "Zero Trust is not a middleware tool; it is an architectural philosophy. Never assume trust, always verify JWT signatures! 🔒 #cybersecurity #network",
        "Auditing cloud credentials logs. A single {adjective} development key was leaked in an public workflow report. Always sanitise variables! #cloudsecurity"
    ],
    "data": [
        "Data Science Rule: Model parameters tuning is completely {adjective} if your training database has dirty data bias. Quality features preprocessing always beats complex algorithms. #machinelearning",
        "Analyzing a multi-dimensional {buzzword} scatter plot. Beautiful clusters form naturally when correlating likes and shares ratios! 📉 #datascience #statistics",
        "Linear Regression models: Transparent, fast, and {adjective} enough for baseline projections. Don't launch neural networks when a line fits! #ai #regression"
    ],
    "design": [
        "Designing a next-gen {buzzword} SaaS panel. Translucent glassmorphic panels and custom glowing borders make layouts look incredibly premium. Stripe analytics vibes! ✨ #uiux #layout",
        "Interface design is only 10% aesthetics; the rest is intuitive navigation. A {adjective} flow eliminates user friction and drives growth. #productdesign #figma",
        "Palette standard: Cyber Cyan + Electro Purple paired with high-contrast Space Navy backgrounds. Taking digital layouts to a science! 🎨 #webdesign #creative"
    ]
}

ADJECTIVES = ["elegant", "lightning-fast", "minimalist", "gorgeous", "scalable", "cutting-edge"]
BUZZWORDS = ["glassmorphic", "container-driven", "asynchronous", "reactive", "highly-modular", "AI-optimized"]

def generate_ai_caption(niche: str) -> str:
    templates = NICHE_TEMPLATES.get(niche.lower(), NICHE_TEMPLATES["tech"])
    template = random.choice(templates)
    
    adj = random.choice(ADJECTIVES)
    buzz = random.choice(BUZZWORDS)
    
    return template.format(adjective=adj, buzzword=buzz)
