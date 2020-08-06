import numpy as np  # импортируем библиотеку для работы с массивами данных
import tensorflow.keras  # импортируем нейросетевую библиотеку
from tensorflow.keras.models import Model, load_model  # из кераса подгружаем абстрактный класс базовой модели,
from tensorflow.keras.layers import Dense, Flatten, Input, Lambda, Conv2D, MaxPooling2D, Reshape, Multiply
from tensorflow.keras.optimizers import RMSprop, Adam  # из кераса загружаем выбранный оптимизатор
from tensorflow.keras import backend as be

class Network(object):
    def __init__(self):
        self.__version__ = version
    
    def createModel(self, rays=5, actions=5):
        inputs = Input(shape=(rays + 3,))  # 3 for V, sin(a), reward
        fullConnected = Dense(units=100, activation='relu', use_bias=False)(inputs)  # задали 100 нейронов
        sigmoidOutput = Dense(actions, activation='sigmoid', use_bias=False)(fullConnected)  # сигмоида на выходе
        policyNetworkModel = Model(inputs=inputs, outputs=sigmoidOutput)  # собрали модель стратегии
        policyNetworkModel.summary()  # посмотрим на модель
        episodeReward = Input(shape=(1,), name='episodeReward')  # задаем награду за эпизод
        self.policyNetworkTrain = Model(inputs=[inputs, episodeReward], outputs=sigmoidOutput)
        myOptimizer = RMSprop(lr=0.0001)  # выбрали оптимизатор с заданной скоростью обучения
        self.policyNetworkTrain.compile(optimizer=myOptimizer, loss=rewardedLoss(episodeReward))
        return self.policyNetworkTrain


def rewardedLoss(reward):  # задаем новую функцию потерь, принимающую episodeReward, награда
    def loss(yTrue, yPred):
        # подаём в кач-ве yTrue фактически сделанное действие(action)
        # если фактически сделанное действие было движением вверх - подаем 1 на yTrue, если нет то подаем 0
        # yPred - выход сетки(вероятность выбора движения вверх)
        # мы не подаём yPred в нейронку, его вычисляет керас

        # сначала log(0) and log(1) неопределены - загоняем yPred между значениями:
        tmpPred = Lambda(lambda x: be.clip(x, 0.05, 0.95))(yPred)
        # вычисляем логарифм вероятности. yPred - вероятность выбора движения вверх
        # помним что yTrue = 1 когда фактически выбрано движение вверх, и 0 - когда вниз
        # формула похожа на кросс-энтропию в керасе, но здесь мы прописываем её вручную,
        # чтобы умножить на значение награды
        tmpLoss = Lambda(lambda x: -yTrue * be.log(x) - (1 - yTrue) * (be.log(1 - x)))(tmpPred)
        # обновленная функция потерь - "функция политики"
        policyLoss = Multiply()([tmpLoss, reward])  # добавляем в loss умножение на награду за эпизод
        return policyLoss  # ввели обновленную функцию политики

    return loss  # возвращаем обновленную функцию политики


version = 'PNT 1.0.2 06.08.2020'
print(version)