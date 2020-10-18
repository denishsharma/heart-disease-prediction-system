from flask import Flask, render_template, request, redirect, send_from_directory, jsonify
from joblib import load
import pandas as pd
import numpy as np
from joblib import dump
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, f1_score

app = Flask(__name__)


model = load('model.joblib')
def predict(features):
   return (int)((model.predict([features]) >= 0.5).astype(int))


@app.route('/')
def index():
   # a = predict([63,1,3,145,233,1,0,150,0,2.3,0,0,1])
   return render_template('index.html')


@app.route('/predict', methods=['post', 'get'])
def predictPage():
   pred = None
   if request.method == 'POST':
      age = int(request.form.get('age'))
      gender = int(request.form.get('gender'))
      chestPain = int(request.form.get('chestPain'))
      rbp = int(request.form.get('rbp'))
      scl = int(request.form.get('scl'))
      fbs = 1 if int(request.form.get('fbs')) > 120 else 0
      restecg = int(request.form.get('restecg'))
      hr = int(request.form.get('hr'))
      exerangina = int(request.form.get('exerangina'))
      st = float(request.form.get('st'))
      slope = int(request.form.get('slope'))
      ca = int(request.form.get('ca'))
      thal = int(request.form.get('thal'))
      
      param = [age, gender, chestPain, rbp, scl, fbs, restecg, hr, exerangina, st, slope, ca, thal]
      print(param)
      pred = predict(param)

   return render_template('predict.html', pred=pred)


@app.route('/downloads')
def downloadPage():
   return render_template('downloads.html')


@app.route('/downloads/<path:filename>', methods=['GET', 'POST'])
def download(filename):    
    return send_from_directory(directory='static/downloads', filename=filename)


@app.route('/sysdis', methods=['get'])
def sysdisPage():
   train_df = pd.read_csv('static/dataset/sysdis/train.csv')
   test_df = pd.read_csv('static/dataset/sysdis/test.csv')

   cols = train_df.columns
   cols = cols[:-1]

   x = train_df.iloc[:, :-1]
   y = train_df['disease']

   le = preprocessing.LabelEncoder()
   le.fit(y)
   y = le.transform(y)

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
   testX = test_df[cols]

   clf = DecisionTreeClassifier()
   clf.fit(x_train, y_train)

   scores = cross_val_score(clf, x_test, y_test, cv=10)
   modelInfo = {
         'classifier': 'DecisionTreeClassifier',
         'score': scores.mean(),
         'std': scores.std()
      }
    
   symptoms = request.args.getlist('symp')
   symptomsCount = len(symptoms)

   symptomsInfo = {
      'symptomCount': symptomsCount,
      'symptoms': symptoms
   }

   predInfo = {}

   # try:
   _symptoms_dict = {}
   for _index, _symptom in enumerate(x):
      _symptoms_dict[_symptom] = _index
   input_vector = np.zeros(len(_symptoms_dict))
   for item in symptoms:
      input_vector[[_symptoms_dict[item]]] = 1
   pred = le.inverse_transform(clf.predict([input_vector]))

   diseasePred = []
   inputVectorParam = []

   for inputVal in input_vector:
      inputVectorParam.append(int(inputVal))

   for disease in pred:
      diseasePred.append(disease)

   predInfo = {
      'status': 2,
      'inputValue': inputVectorParam,
      'prediction': diseasePred,
      'totalInputCount': len(inputVectorParam)
   }
   # except:
   #   predInfo = {'status': 1}



   resp = {**modelInfo, **symptomsInfo, **predInfo}

   # return jsonify(request.args.getlist('symp[]'))
   return jsonify(resp)


@app.route('/sysdis/symptom/all', methods=['get'])
def sysdisSymptomAllPage():
   df = pd.read_csv('static/dataset/sysdis/test.csv')
   x = df.iloc[:, :-1]
   symps = []
   for _index, _symptom in enumerate(x):
      symps.append({
         'id': int(_index),
         'name': _symptom
      })

   resp = {
      'count': len(symps),
      'symptoms': symps
   }
   return jsonify(resp)



@app.route('/sysdis/disease/all', methods=['get'])
def sysdisDiseaseAllPage():
   df = pd.read_csv('static/dataset/sysdis/train.csv')
   x = df['disease'].unique()
   diseases = []
   for _index, _disease in enumerate(x):
      diseases.append({
         'id': int(_index),
         'name': _disease
      })

   resp = {
      'count': len(diseases),
      'diseases': diseases
   }
   return jsonify(resp)




@app.errorhandler(404)
def notFound(e):
   return render_template('404.html')



if __name__ == '__main__':
    app.run(debug=True)
