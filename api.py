from flask import Flask, request, jsonify
# Cross-Origin Resource Sharing (CORS)
# Modern browsers apply the "same-origin policy", which blocks web pages from
# making requests to a different origin than the one that served the page.
# This helps prevent malicious sites from reading sensitive data from another
# site you are logged into.
#
# However, there are many legitimate cases where cross-origin requests are
# needed. One example is:
#
## Single-Page Applications (SPA) hosted at example-frontend.com need to call
## APIs hosted at api.example-backend.com.
#
# To support this safely, CORS lets servers explicitly allow such requests.
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
# CORS(
#     app,
#     resources={r"/api/*": {
#         "origins": [
#             "https://127.0.0.1",
#             "https://localhost"
#         ]
#     }},
#     methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type"]
# )

CORS(
    app, supports_credentials=False,
    resources={r"/api/*": { # This means CORS will only apply to routes that start with /api/
               "origins": [
                   "https://127.0.0.1", "https://localhost",
                   "https://127.0.0.1:443", "https://localhost:443",
                   "http://127.0.0.1", "http://localhost",
                   "http://127.0.0.1:5000", "http://localhost:5000",
                   "http://127.0.0.1:5500", "http://localhost:5500"
                   "https://refactored-carnival-56jrvpj6q96fpxq9-5500.app.github.dev",
                   "https://refactored-carnival-56jrvpj6q96fpxq9-5000.app.github.dev"
                ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"])

CORS(app, supports_credentials=False,
     origins=["*"])

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')

naive_bayes_classifier = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')
knn_classifier = joblib.load('./model/knn_classifier_optimum.pkl')
svm_classifier = joblib.load('./model/support_vector_classifier_optimum.pkl')
random_forest_classifier = joblib.load('./model/random_forest_classifier_optimum.pkl')
# Load preprocessors for the 4 classifiers
label_encoders_2 = joblib.load('./model/label_encoders_2.pkl')
label_encoders_4 = joblib.load('./model/label_encoders_4.pkl')
label_encoders_5 = joblib.load('./model/label_encoders_5.pkl')
scaler_4 = joblib.load('./model/scaler_4.pkl')
scaler_5 = joblib.load('./model/scaler_5.pkl')

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "ML API is running"}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

# Defines an HTTP endpoint
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    # Accepts JSON data sent by a client (browser, curl, Postman, etc.)
    data = request.get_json()
    # Create a DataFrame with the correct feature names
    new_data = pd.DataFrame([{
        'monthly_fee': data.get('monthly_fee'),
        'customer_age': data.get('customer_age'),
        'support_calls': data.get('support_calls')
    }])

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'monthly_fee',
        'customer_age',
        'support_calls'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Performs a prediction using the already trained machine learning model
    prediction = decisiontree_classifier_baseline.predict(new_data)[0]
    
    # Returns the result as a JSON response:
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

# *2.a.* Sample cURL POST values (without HTTPS in NGINX and Gunicorn)

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *3* Sample PowerShell values:

# $body = @{
#     monthly_fee = 60
#     customer_age = 30
#     support_calls = 1
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'])
def predict_naive_bayes_classifier():
    data = request.get_json()
    
    # Create DataFrame with all 17 features
    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])
    
    # Encode categorical features
    for col in ['VisitorType', 'Weekend', 'Month']:
        if col in new_data.columns:
            new_data[col] = label_encoders_2[col].transform(new_data[col])
    
    # Scale features
    new_data_scaled = scaler_4.transform(new_data)
    
    # Predict
    prediction = naive_bayes_classifier.predict(new_data_scaled)[0]
    
    return jsonify({'Predicted Class': int(prediction)})

@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn_classifier():
    data = request.get_json()
    
    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])
    
    for col in ['VisitorType', 'Weekend', 'Month']:
        if col in new_data.columns:
            new_data[col] = label_encoders_4[col].transform(new_data[col])
    
    new_data_scaled = scaler_4.transform(new_data)
    prediction = knn_classifier.predict(new_data_scaled)[0]
    
    return jsonify({'Predicted Class': int(prediction)})

@app.route('/api/v1/models/svm-classifier/predictions', methods=['POST'])
def predict_svm_classifier():
    data = request.get_json()
    
    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])
    
    for col in ['VisitorType', 'Weekend', 'Month']:
        if col in new_data.columns:
            new_data[col] = label_encoders_5[col].transform(new_data[col])
    
    new_data_scaled = scaler_5.transform(new_data)
    prediction = svm_classifier.predict(new_data_scaled)[0]
    
    return jsonify({'Predicted Class': int(prediction)})

@app.route('/api/v1/models/random-forest-classifier/predictions', methods=['POST'])
def predict_random_forest_classifier():
    data = request.get_json()
    
    new_data = pd.DataFrame([{
        'Administrative': data.get('Administrative'),
        'Administrative_Duration': data.get('Administrative_Duration'),
        'Informational': data.get('Informational'),
        'Informational_Duration': data.get('Informational_Duration'),
        'ProductRelated': data.get('ProductRelated'),
        'ProductRelated_Duration': data.get('ProductRelated_Duration'),
        'BounceRates': data.get('BounceRates'),
        'ExitRates': data.get('ExitRates'),
        'PageValues': data.get('PageValues'),
        'SpecialDay': data.get('SpecialDay'),
        'Month': data.get('Month'),
        'OperatingSystems': data.get('OperatingSystems'),
        'Browser': data.get('Browser'),
        'Region': data.get('Region'),
        'TrafficType': data.get('TrafficType'),
        'VisitorType': data.get('VisitorType'),
        'Weekend': data.get('Weekend')
    }])
    
    for col in ['VisitorType', 'Weekend', 'Month']:
        if col in new_data.columns:
            new_data[col] = label_encoders_5[col].transform(new_data[col])
    
    new_data_scaled = scaler_5.transform(new_data)
    prediction = random_forest_classifier.predict(new_data_scaled)[0]
    
    return jsonify({'Predicted Class': int(prediction)})

# *1* Sample JSON POST values
# {
#     "CustomerType": "Business",
#     "BranchSubCounty": "Kilimani",
#     "ProductCategoryName": "Meat-Based Dishes",
#     "QuantityOrdered": 8,
#     "PaymentDate": "2027-11-13"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *2.b.* Sample cURL POST values

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *3* Sample PowerShell values:

# $body = @{
#     PaymentDate         = "2027-11-13"
#     CustomerType        = "Business"
#     BranchSubCounty     = "Kilimani"
#     ProductCategoryName = "Meat-Based Dishes"
#     QuantityOrdered = 8
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

# This ensures the Flask web server only starts when you run this file directly
# (e.g., `python api.py`), and not if you import api.py from another script or test.

# __name__ is a special variable in Python. When you run a script directly,
# __name__ is set to '__main__'. If the script is imported, __name__ is set to
# the module's name.

# if __name__ == '__main__': checks if the script is being run directly.

# app.run(debug=True) starts the Flask development server with debugging enabled.
# This means:
## The server will automatically reload if you make code changes.
## You get detailed error messages in the browser if something goes wrong.

@app.route('/api/v1/models/recommender/predictions', methods=['POST'])
def recommend_products():
    data = request.get_json()
    cart = data.get('cart', [])
    
    if not cart:
        return jsonify({
            'error': 'Cart is empty',
            'recommendations': []
        }), 400
    
    cart_set = set(cart)
    recommendations = []
    
    # Simple rule-based recommendations
    if 'whole milk' in cart_set:
        recommendations = ['other vegetables', 'rolls/buns', 'yogurt']
    elif 'yogurt' in cart_set:
        recommendations = ['whole milk', 'tropical fruit']
    else:
        recommendations = ['whole milk', 'other vegetables']
    
    # Remove items already in cart
    recommendations = [item for item in recommendations if item not in cart_set]
    
    return jsonify({
        'recommendations': recommendations[:5],
        'cart_items': cart
    })

@app.route('/api/v1/models/cluster-predictor/predictions', methods=['POST'])
def predict_cluster():
    data = request.get_json()
    age = data.get('age')
    annual_income = data.get('annual_income')
    spending_score = data.get('spending_score')
    
    if not all([age, annual_income, spending_score]):
        return jsonify({
            'error': 'Missing required fields: age, annual_income, spending_score'
        }), 400
    
    # Simple rule-based cluster prediction
    if spending_score > 70:
        cluster = 0
        description = "High spenders"
    elif annual_income > 60:
        cluster = 1
        description = "High income, moderate spending"
    elif age < 30:
        cluster = 2
        description = "Young, budget-conscious"
    else:
        cluster = 3
        description = "Mature, balanced profile"
    
    return jsonify({
        'predicted_cluster': cluster,
        'cluster_description': description,
        'input_features': {
            'age': age,
            'annual_income': annual_income,
            'spending_score': spending_score
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)