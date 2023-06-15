import os
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from dotenv import load_dotenv
import shutil

#models
from catboost import Pool ,CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import pickle

app = Flask(__name__)

load_dotenv()
# Получаем параметры из .env файла
model_dir = os.environ.get('MODEL_DIR')
num_cores = int(os.environ.get('NUM_CORES',1))
max_models = int(os.environ.get('MAX_MODELS',1))

print(model_dir,num_cores,max_models)

#HELLO WORLD
@app.route('/', methods=['GET'])
def hello_world():
   return 'Hello, World!'
@app.route('/post')
def hello_path():
   return 'Hello, Path!'


# Модельки и все остальное
@app.route("/fit", methods=['POST'])
def fit():
    input_json = request.get_json()
    config = input_json['config']
    model_name = config["model_name"]
    model_type = config['model_type']

    X = input_json['X']
    y = input_json['y']
    
    train_log = ""
    #train
    if model_type == "CatboostRegressor" :
        train_log += "CatboostRegressor fitting :\n"
        X = np.array(X)
        y = np.array(y)
        train_data = Pool(data=X, label=y)
        model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6)
        model.fit(train_data)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model.save_model(os.path.join(model_dir, model_name))
        train_log += "Success"
    elif model_type == "sklearn.linear_model":
        train_log += "\nSklearn linear model fitting :\n"
        X = np.array(X)
        y = np.array(y)
        model = LinearRegression()
        model.fit(X, y)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        joblib.dump(model, model_dir + '/' + model_name )
        train_log += "Success"
    else :
        return {'message': 'this type model dont exists' , "train_log" : train_log}

    return {"message": f"Model '{model_name}' trained and saved" ,
            "train_log" : train_log
            }

@app.route('/predict', methods=['POST'])
def predict():
    input_json = request.get_json()
    X_test = input_json['X_test']
    config = input_json['config']
    model_type = config['model_type']
    model_name = config["model_name"]
    if model_type == "CatboostRegressor" :
        loaded_model = CatBoostRegressor()
        loaded_model.load_model(model_dir + '/' + model_name )
        result = loaded_model.predict(X_test)
    elif model_type == "sklearn.linear_model":
        loaded_model = LinearRegression()
        loaded_model.load_model(model_dir + '/' + model_name )
        result = loaded_model.predict(X_test)
    else :
        return {'message': 'this type model dont exists' }
    return {'predict' : list(result)}
            
@app.post("/load")
def load():
    input_json = request.get_json()
    config = input_json['config']
    model_type = config['model_type']
    model_name = config["model_name"]
    decoded_model = input_json['decoded_model']
    model_path = os.path.join(model_dir, f"{model_name}")
    if os.path.exists(model_path):
        return {"error": f"Model '{model_name}' already exists"}
    if model_type == "CatboostRegressor" :
        model = pickle.loads(decoded_model.encode('latin1'))
        model.save_model(os.path.join(model_dir, model_name))
    elif model_type == "sklearn.linear_model":
        model = pickle.loads(decoded_model.encode(latin1))
        joblib.dump(model, model_dir + '/' + model_name )
    else :
        return {'message': 'this type model dont exists' }


    return {"message": f"Model '{model_name}' loaded for inference"}

@app.post("/unload")
def unload():
    input_json = request.get_json()
    config = input_json['config']
    model_type = config['model_type']
    model_name = config["model_name"]
    model_path = os.path.join(model_dir, f"{model_name}")
    
    if model_type == "CatboostRegressor" :
        model = CatBoostRegressor()
        model.load_model(model_path)
        decoded_model = pickle.dumps(model).decode("latin1")
        result = decoded_model   
    elif model_type == "sklearn.linear_model":
        model = LinearRegression()
        model.load_model(model_path)
        decoded_model = pickle.dumps(model).decode("latin1")
        result = decoded_model
    else :
        return {'message': 'this type model dont exists' }
    return {"decoded_model": result , "message": f"Model '{model_name}' unloaded"}

@app.post("/remove")
def remove():
    input_json = request.get_json()
    config = input_json['config']
    model_name = config["model_name"]
    model_path = os.path.join(model_dir, f"{model_name}")
    os.remove(model_path)
    return {"message": f"Model '{model_name}' removed"}

@app.post("/remove_all")
def remove_all():
    shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return {"message": "All models removed"}

if __name__ == '__main__':
    app.run()