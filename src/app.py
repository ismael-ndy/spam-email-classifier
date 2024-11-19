from flask import Flask, request, jsonify, render_template
from app_utils import spam_classification

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        email = request.form.get('email')
        model_name = request.form.get('model')

        prediction = spam_classification(email, model_name)

        if prediction == "spam":
            return jsonify({'result': 'spam'})
        else:
            return jsonify({'result': 'not spam', 'response': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)