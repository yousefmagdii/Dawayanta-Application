import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model

# necessary imports
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import csv
import random
from math import floor
import argparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
###
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer





#------------------------------------------------------------------------------------------------------

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
FinalDS_Path = 'Dataset/Final_Dataset.csv'
def dataset_creation():
    disease_symptom_dict = defaultdict(list)
    with open(FinalDS_Path,'r',encoding='utf-8') as csvFile:
      reader=csv.reader(csvFile)
      data=list(reader)
    csvFile.close()
    keywords=[]

    for i in range(1,len(data)):
      data[i][0].replace('\xa0',' ')
      for j in range(0,len(data[i][1])):
        data[i][1][j].replace('\xa0',' ')
      keywords=data[i][1].split(',')
      disease_symptom_dict[str(data[i][0]).strip()].append(keywords)
    x = []
    y = []
    for i in disease_symptom_dict:
        y.append(i)
        x.append(disease_symptom_dict[i][0])
    temp = []
    for i in x:
        for j in i:
            temp.append(j)

    from collections import OrderedDict
    temp = list(OrderedDict((x, True) for x in temp).keys())
    for i in range(len(y)):
      for j in range(150):
          y.append(y[i])
          x.append(random.sample(x[i],floor(len(x[i])/3)))
    x_ = []
    for i in x:
      x_.append(' '.join(i))
    return x_,y
    
def init():
    global model, classifiers, training_data, final_y, count_vector, data_y, onehotencoder, le

    # load the pre-trained Keras model
    # this function helps in dataset creation by creating a disease-symptom dictionary
    #-------------------Dataset Creation-------------------
    
    
    
    # now data_x,data_y contains the diseases and their respective symptoms 
    data_x,data_y=dataset_creation()
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2,random_state=0)

    #-------------------Encoders-------------------
    le = LabelEncoder()
    output_data_complete= le.fit_transform(data_y)
    output_data=le.transform(y_train)
    output_data_test=le.transform(y_test)
    onehotencoder = OneHotEncoder()
    output_data_complete=onehotencoder.fit_transform(output_data_complete.reshape(-1,1))
    output_data=onehotencoder.transform(output_data.reshape(-1,1))
    output_data_test=onehotencoder.transform(output_data_test.reshape(-1,1))

    #-------------------Vectorizer-------------------
    count_vector = CountVectorizer(max_features=300)  
    training_data = count_vector.fit_transform(x_train) 
    final_y=onehotencoder.transform((le.transform(y_train)).reshape(-1,1))


    #-------------------Load and Fit Model-------------------
    model = load_model('Checkpoints/my_model.h5')
    classifiers=[model]
    
#    graph = tf.get_default_graph()

def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('symptoms'))

#    parameters.append(flask.request.args.get('full_accuracy'))
    return parameters

# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response


# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    nameOfTheCharacter = flask.request.args.get('name')
    parameters = getParameters()
    for h in range(0,len(classifiers)):

        classifiers[h].fit(training_data.todense(),final_y.toarray(),epochs=20, batch_size = 50)

    print([flask.request.args.get('symptoms')])
    Ftest = [flask.request.args.get('symptoms')]
    testing_data = count_vector.transform(Ftest)[0]

    ev = []
    data_y_nd = list(dict.fromkeys(data_y))

    for d in data_y_nd:
      Y_data = onehotencoder.transform((le.transform([d])).reshape(-1,1))
      data_evaluate = model.evaluate(testing_data.todense(), Y_data.toarray(), verbose=0)
      ev.append([d, data_evaluate])

    new_ev = copy.deepcopy(ev)

    from operator import itemgetter
    new_ev = sorted(new_ev, key=itemgetter(1, 0))

    f_ev = copy.deepcopy(new_ev[:5])

    sum_all = sum(i[1][0] for i in f_ev)

    max_acc = max(f_ev, key=lambda x: x[1][0])[1][0]

    sum0 = 0
    for f in f_ev:
      f[1][0] = (max_acc - f[1][0])


    sum_all = sum(i[1][0] for i in f_ev)

    for f in f_ev:
        f[1][0] = (f[1][0] / sum_all) * 100
        sum0 += f[1][0]

#    print(f_ev[:4])
    return sendResponse({nameOfTheCharacter: f_ev[:4]})

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)

