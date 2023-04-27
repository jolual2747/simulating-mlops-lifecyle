from flask import Flask, request, jsonify
from predicting import make_prediction


app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({'message' : 'Hello, World!'})

@app.route('/predict', methods = ['POST'])
def get_prediction():
    features = request.get_json()
    print(type(features))
    pred, prob = make_prediction(features)
    return jsonify({"class": str(pred), "prob":  prob})

if __name__ == '__main__':
    app.run(debug=True, port=5000)