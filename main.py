from flask import Flask, render_template, request
import pickle
import yaml
import os

import pandas as pd
from training import Train



app = Flask(__name__)


if os.path.exists('config.yaml'):
    with open('config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
else:
    raise FileNotFoundError("Config file not found")


train_instance = Train(configs)
model_pipeline, x_train, x_test, y_train, y_test = train_instance.train_model()

with open("model.pkl", 'wb') as f:
    pickle.dump(model_pipeline, f)

with open("model.pkl", 'rb') as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            User_ID = request.form.get('User_ID')
            Gender = request.form.get('Gender')
            Age = request.form.get('Age')
            EstimatedSalary = request.form.get('EstimatedSalary')

            input_data = pd.DataFrame([[
                User_ID, Gender, Age, EstimatedSalary
            ]], columns=configs['input_feature'])

            print("Input Data:\n", input_data)
            input_data = input_data.astype({
                'User_ID': int,
                'Age': int,
                'Gender':object,
                'EstimatedSalary': int
            })
            print("Converted Input Data:\n", input_data)

            result = model_pipeline.predict(input_data)[0]
            print('Model Prediction:', result)


            return render_template("result.html", result=result)
        except Exception as e:
            print("Error during prediction:", e)
            return render_template("index.html", error=str(e))
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
