from Backprop import *


def predictUser(data):
    fileTrain = "https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/DataTrainApi.csv"
    dftrainfix = devideData(80, fileTrain)
    Xtrain = dftrainfix.iloc[:, :-1]
    Ytrain = dftrainfix.iloc[:, -1]
    X_train = minmax_scale(Xtrain)
    y_train = onehot_enc(Ytrain)

    w, ep, mse = bp_fit(X_train, y_train, layer_conf=(
        45, 7, 4), learn_rate=.01, max_epoch=100000, max_error=.1, print_per_epoch=500)
    print(f'Epochs: {ep}, MSE: {mse}')

    predict = bp_predict(normalisasi(data), w)
    predict = onehot_dec(predict)

    return predict


def predictAdmin(jmlDataTrain, jmlDataTest, lRate, nHidden):
    fileTrain = "https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/DataTrainApi.csv"
    fileTest = "https://raw.githubusercontent.com/rizalanhari/ADHD-API/main/DataTestApi.csv"

    dftrainfix = devideData(jmlDataTrain, fileTrain)
    Xtrain = dftrainfix.iloc[:, :-1]
    Ytrain = dftrainfix.iloc[:, -1]
    X_train = minmax_scale(Xtrain)
    y_train = onehot_enc(Ytrain)

    dftestfix = devideData(jmlDataTest, fileTest)
    Xtest = dftestfix.iloc[:, :-1]
    Ytest = dftestfix.iloc[:, -1]
    X_test = minmax_scale(Xtest)
    y_test = onehot_enc(Ytest)

    w, ep, mse = bp_fit(X_train, y_train, layer_conf=(
        45, nHidden, 4), learn_rate=lRate, max_epoch=100000, max_error=.1, print_per_epoch=500)
    print(f'Epochs: {ep}, MSE: {mse}')

    predict = bp_predict(X_test, w)
    predict = onehot_dec(predict)
    y_test2 = onehot_dec(y_test)
    acc = accuracy_score(predict, y_test2)

    return predict, y_test2, acc
