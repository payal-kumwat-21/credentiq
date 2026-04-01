def credibility_score(text):
    text = text.lower()

    # Strong fake indicators
    strong_fake = [
        "whatsapp", "forwarded", "miracle", "cure",
        "secret", "leaked", "hoax", "rumor"
    ]

    # Weak fake indicators (small penalty)
    weak_fake = [
        "viral", "claims", "unverified"
    ]

    # Trusted news signals
    trusted_sources = [
        "pib", "press information bureau",
        "rbi", "reserve bank of india",
        "ministry", "who", "reuters", "bbc",
        "times of india", "according to"
    ]

    # Journalism-style neutral words (no penalty)
    neutral_words = [
        "reports", "suggest", "considering", "may announce",
        "official", "statement", "sources"
    ]

    score = 0

    for word in strong_fake:
        if word in text:
            score -= 0.35

    for word in weak_fake:
        if word in text:
            score -= 0.15

    for source in trusted_sources:
        if source in text:
            score += 0.3

    for neutral in neutral_words:
        if neutral in text:
            score += 0.1  # small positive instead of penalty

    return max(min(score, 0.6), -0.6)

"""def credibility_score(text):
    text = text.lower()

    fake_indicators = [
        "viral", "leaked", "whatsapp", "forwarded",
        "social media", "claims", "rumor", "unverified"
    ]

    trusted_sources = [
        "pib", "press information bureau",
        "rbi", "reserve bank of india",
        "ministry", "who", "reuters", "bbc", "times of india"
    ]

    score = 0

    for word in fake_indicators:
        if word in text:
            score -= 0.25

    for source in trusted_sources:
        if source in text:
            score += 0.3

    return max(min(score, 1), -1)"""
