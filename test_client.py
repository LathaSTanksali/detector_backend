import requests

url = "http://127.0.0.1:5000/api/classify"
data = {"url": "http://paypal.com.secure-login.verify-account.com"}  # fake phishing-looking URL

response = requests.post(url, json=data)
result = response.json()

print("Prediction:", "Phishing" if result["prediction"] == 1 else "Legitimate")
print("Key indicators:", result.get("top_features", []))

# Fetch dashboard stats
stats_response = requests.get("http://127.0.0.1:5000/api/stats")
stats = stats_response.json()
print("Stats so far:", stats)
