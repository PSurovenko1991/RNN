import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding # Embedding - слой для векторного представления
from keras.layers import LSTM #рекурентный слой
from keras.datasets import imdb

np.random.seed(42)


# Указываем максимальное количество слов в тексте:
max_features = 5000
(x_train,y_train),(x_test,y_test) = imdb.load_data(nb_words =max_features)


# Устанавливаем максимальную длинну рецензии:
maxLen = 80
#заполняем пробелами или обрезаем статьи:
x_train = sequence.pad_sequences(x_train,maxlen=maxLen)
x_test = sequence.pad_sequences(x_test,maxlen=maxLen)






#загрузка модели и весов:
from keras.models import model_from_json
jesfile = open("RNN.json","r")
lmj = jesfile.read()
jesfile.close()

Lmodel = model_from_json(lmj)
#Загрузка весов:
Lmodel.load_weights("RNNmodel.h5")


#После загрузки необходима компиляция:
Lmodel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


score2 = Lmodel.evaluate(x_test,y_test)
print("точность загруженной модели: ", score2[1]*100)


