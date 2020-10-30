import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
#import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

# +
raw_data = pd.read_csv('data.csv')
dataset = raw_data.copy()
dataset['diagnosis']=dataset['diagnosis'].map({'B':0,'M':1})
raw_x = dataset.iloc[:, 2:31].values
raw_y = dataset.iloc[:, 1].values

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(learning_rate_init = 0.001, max_iter = 200, momentum = 0.0)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit(raw_x).transform(raw_x)

from imblearn.over_sampling import SMOTE
sampling = SMOTE(random_state=0)
x_train , y_train = sampling.fit_resample(x, raw_y)

#Fitting model with trainig data
model.fit(x_train, y_train)


# -

@app.route('/')
def home():
    return render_template('index.html', variables=None)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    output = {}
    if model.predict(final_features) == 0:
        output["Benign"] = model.predict_proba(final_features)
    elif model.predict(final_features) == 1:
        output["Malignant"] = round(model.predict_proba(final_features)[0][1]*100, 2)
    else :
        output['false'] = model.predict(final_features)
    return render_template('index.html', variables = output)


if __name__ == "__main__":
    app.run(debug=False)




