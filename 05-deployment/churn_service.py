import pickle
from flask import Flask
from flask import request
from flask import jsonify

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)


app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    result = {
        'churn_probability' : float(prediction),
        'churn' : bool(churn)
    }
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)


