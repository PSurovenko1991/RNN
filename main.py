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


#создаем сеть:
model = Sequential()

#добавляем слои:
#слой для векторного представления:
model.add(Embedding(max_features, 32, dropout = 0.2)) # 5000 слов, преобразуются в вектор 32 входа
#слой долго/краткосрочной памяти:
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2)) # w -входные связи, u - рекурентные связи
#полносвязный слой для классификации:
model.add(Dense(1,activation="sigmoid"))



# Компилируем сеть:
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Обучааем сеть:

model.fit(x_train, y_train, batch_size=64, nb_epoch = 7, validation_data=(x_test, y_test), verbose=1)


# проверка точности:
score = model.evaluate(x_test,y_test)
print("Точность: ", score[1]*100)


# Работа модели:
pr = model.predict(x_test)
print(pr[:10].round(0))
print(y_test[:10])




#Сохранение модели:
model_j = model.to_json()
jf = open("RNN.json","w")
jf.write(model_j)
jf.close()

# Сохранение весов:
model.save_weights("RNNmodel.h5")
