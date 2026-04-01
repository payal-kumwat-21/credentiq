import joblib
from utils.preprocessing import clean_text
from utils.credibility import credibility_score
from utils.source_verify import verify_sources

# 1. Load the saved ML files
# Note: In Colab, make sure paths match your current working directory
model = joblib.load("ml/model.pkl")
vectorizer = joblib.load("ml/vectorizer.pkl")

def predict_news(text):
    reasons = []

    # 2. Machine Learning Prediction
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    ml_prob = model.predict_proba(vec)[0][1]

    if ml_prob > 0.7:
        reasons.append("ML model strongly supports this claim.")
    elif ml_prob < 0.4:
        reasons.append("ML model detected linguistic patterns common in fake news.")
    else:
        reasons.append("ML model confidence is moderate.")

    # 3. Rule-Based Credibility Score
    cred_score = credibility_score(text)
    if cred_score < 0:
        reasons.append("Contains terms frequently used in misinformation.")
    elif cred_score > 0:
        reasons.append("Contains language commonly found in credible news.")

    # 4. API Source Verification
    # We grab the first 5 words instead of 100 characters to make a cleaner API query
    query = " ".join(text.split()[:5])
    sources = verify_sources(query)

    if sources:
        reasons.append("Claim is reported by trusted news sources.")
    else:
        reasons.append("No trusted news sources found reporting this claim.")

    # 5. Final Score Calculation
    source_boost = 0.25 if sources else 0
    final_score = ml_prob + cred_score + source_boost
    final_score = max(min(final_score, 1.0), 0.0) # Keep score between 0 and 1

    # 6. Assign Label
    if final_score >= 0.7:
        label = "Real ✅"
    elif final_score >= 0.45:
        label = "Suspicious ⚠️"
    else:
        label = "Fake ❌"

    return label, round(final_score, 2), sources, reasons
