from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

pipe = pickle.load(open('pipe.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    company = request.form['company']
    typename = request.form['typename']
    ram = int(request.form['ram'])
    weight = float(request.form['weight'])
    opsys = request.form['opsys']
    cpu = request.form['cpu']
    gpu = request.form['gpu']
    memory = request.form['memory']

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'Ram': [ram],
        'Weight': [weight],
        'OpSys': [opsys],
        'Cpu': [cpu],
        'Gpu': [gpu],
        'Memory': [memory]
    })

    prediction = pipe.predict(query)[0]

    prediction_inr = prediction * 90

    return render_template(
        'index.html',
        prediction_text=f"Estimated Price: ₹ {round(prediction_inr, 2)}"
    )


if __name__ == '__main__':
    app.run(debug=True)