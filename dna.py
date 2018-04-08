import numpy as np
import csv


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten

classes  = 1
iterdata = []
labels = []
LRate = 0.003

def extract_data(filename, data):
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
    for j in range(len(rows)):
        k = 0
        iterdata = []
        if filename == 'train.csv':
            if rows[j][2] == '1':
                labels.append([1])
            else:
                labels.append([0])
        for i in rows[j][1][:]:
            if i == "A":
                iterdata.append([1,0,0,0])
            elif i == "C":
                iterdata.append([0,1,0,0])
            elif i == "G":
                iterdata.append([0,0,1,0])
            elif i == "T":
                iterdata.append([0,0,0,1])
            k+=1
        data.append(iterdata)

    return data, labels
def model():
    
    #First Hidden layer (Input going to Hidden)
    Nnet = Sequential([Dense(12, input_dim=56)])
    
    #Second Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(24, activation='relu'))
    
    #Third Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(48, activation='relu'))
    
       #Third Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(96, activation='relu'))
    
        #Third Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(48, activation='relu'))
    
    #Fourth Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(24, activation='relu'))
    
    #Fifth Hidden layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(12, activation='relu'))
    
    #Sixth Output Layer
    Nnet.add(Dropout(0.03))
    Nnet.add(Dense(classes, activation='sigmoid'))
    
    #mean square error

    Nnet.compile(loss='mse', optimizer=Adam(lr=LRate), metrics=['accuracy'])
                 
    return Nnet

if __name__ == "__main__":
    data = []
    testdata = []
    data, labels  = extract_data("train.csv", data)

    trainData = (np.array(data)).reshape(2000, 56)
    testdata, _  = extract_data("test.csv", testdata)
    testData = (np.array(testdata)).reshape(400, 56)

    model = model()
    labels = np.array(labels)



    model.fit(trainData, labels, batch_size=32, epochs=150)
    scores = model.evaluate(trainData, labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    prediction = model.predict(testData)
    prediction = prediction.reshape(400)
    pred = []
    id = []
    x = 0
    for pre in prediction:
        pre = (np.round(pre)).astype(np.int64)
        pred.append(pre)
        id.append(x)
        x += 1

    print(type(prediction[1]))
    with open("submit.csv","w", newline='') as csvfile:
        wr = csv.writer(csvfile)
        i = 0
        wr.writerow(('id','prediction'))
        for pre in pred:
            wr.writerow((i, pre))
            i+=1



