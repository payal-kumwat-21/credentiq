import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predict import predict_news

# -----------------------------
# Test Case Format:
# (Test ID, Input Text, Expected Label)
# Expected Label: "Real" or "Fake"
# -----------------------------

TEST_CASES = [
    ("TC-01", "The Reserve Bank of India announced new measures to improve liquidity in the banking sector, according to a press release published on its official website.", "Real"),
    ("TC-02", "The World Health Organization released updated vaccination guidelines to help prevent the spread of seasonal influenza worldwide.", "Real"),
    ("TC-03", "According to Reuters, global oil prices rose after OPEC confirmed production cuts during its latest meeting.", "Real"),

    ("TC-04", "Scientists discovered a secret underground city on Mars where humans lived thousands of years ago, revealed by leaked space agency files.", "Fake"),
    ("TC-05", "A WhatsApp message claims that drinking boiled mango leaves can cure diabetes permanently in just 10 days.", "Fake"),
    ("TC-06", "A viral video shows a man who claims he can charge a mobile phone using only his body energy.", "Fake"),

    ("TC-07", "A leaked government report reveals that all internet services will be permanently shut down every Sunday across the country.", "Fake"),
    ("TC-08", "A viral social media post claims the Indian government will make all digital payments completely free starting next month.", "Fake"),
    ("TC-09", "An unverified blog claims that the Ministry of Finance will cancel all personal loans issued before 2020.", "Fake"),

    ("TC-10", "Reports suggest a major technology company may announce a new artificial intelligence device later this year, though no official confirmation has been given.", "Real"),
    ("TC-11", "Social media users speculate that a new tax reform policy might be introduced after the upcoming parliamentary session.", "Fake"),

    ("TC-12", "The Press Information Bureau confirmed that new railway safety guidelines will be implemented starting April, according to an official statement.", "Real"),
    ("TC-13", "The BBC reported that global leaders met to discuss climate change commitments at the international summit held this week.", "Real"),

    ("TC-14", "A viral post referencing Reuters claims the government will provide free laptops to all citizens without any official press release.", "Fake"),
    ("TC-15", "According to local news reports, authorities are considering new traffic regulations, but no official government notification has been issued yet.", "Real"),
]

# -----------------------------
# Test Runner
# -----------------------------

def run_tests():
    total = len(TEST_CASES)
    passed = 0
    fake_correct = 0
    real_correct = 0
    fake_total = 0
    real_total = 0

    print("\n🧪 CredentIQ Automated Test Suite\n" + "-" * 50)

    for tc_id, text, expected in TEST_CASES:
        label, confidence, _, _ = predict_news(text)

        predicted = "Real" if "Real" in label else "Fake"

        result = "PASS ✅" if predicted == expected else "FAIL ❌"

        print(f"\n{tc_id}")
        print(f"Input: {text[:80]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Confidence: {confidence}")
        print(f"Result: {result}")

        if predicted == expected:
            passed += 1
            if expected == "Fake":
                fake_correct += 1
            else:
                real_correct += 1

        if expected == "Fake":
            fake_total += 1
        else:
            real_total += 1

    print("\n" + "-" * 50)
    print("📊 Final Evaluation Report")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Accuracy: {round((passed / total) * 100, 2)}%")

    if fake_total > 0:
        print(f"Fake Detection Rate: {round((fake_correct / fake_total) * 100, 2)}%")

    if real_total > 0:
        print(f"Real Detection Rate: {round((real_correct / real_total) * 100, 2)}%")

    print("\n🏁 Test Run Completed\n")

# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":
    run_tests()
