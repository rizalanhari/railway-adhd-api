from flask import Flask, jsonify, request
import os
import pandas as pd
import json
from BackpropData import *

app = Flask(__name__)

dataTrain = pd.read_csv("https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/DataTrainApi.csv")
dataTest = pd.read_csv("https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/DataTestApi.csv")
question = pd.read_csv("https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/question.csv")

@app.route('/')
def getHello():
    return "Hello, API ADHD."

@app.route('/datatrain')
def getDataTrain():
    return dataTrain.to_json(orient='records')

@app.route('/datatest')
def getDataTest():
    return dataTest.to_json(orient='records')


@app.route('/question')
def getDataQuestion():
    return question.to_json(orient='records')

@app.route('/predictA')
def predictA():
    train = int(request.args.get('train'))
    test = int(request.args.get('test'))
    lrate = float(request.args.get('lrate'))
    neuronh = int(request.args.get('neuronh'))
    output, true, acc = predictAdmin(train, test, lrate, neuronh)
    print("Train: ", train)
    print("Test: ", test)
    print("nHidden: ", neuronh)
    print("lrate: ", lrate)
    print("Acc: ", acc)
    output2 = np.append(output, acc)
    true2 = np.append(true, acc)
    heading = []
    cols = len(output)
    for i in range(cols):
        heading.append('data %d' % (i+1))
    heading.append("acc")
    result = np.vstack((output2, true2))
    df = pd.DataFrame(result, columns=heading)
    return df.to_json(orient='records')


@app.route('/predict')
def predict():
    data = []
    col = []
    for x in range(45):
        temp_data = int(request.args.get('data'+str(x)))
        print(temp_data)
        col.append(temp_data)
    data.append(col)
    hasil = predictUser(data)
    json_hasil = json.dumps({'Hasil': int(hasil[0])})
    return json_hasil


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
