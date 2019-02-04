import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Surface
from plotly.graph_objs import Scatter3d


class Throttle:

    def __init__(self, data, resultDirectory, speedGroupNum=10, throttleGroupNum=10, speedRange=20, throttleRange=1.0,
                 speedInterval=0.1, throttleInterval=0.01, train=True, save=True):
        self.dfPacmod = data.copy()
        self.resultDirectory = resultDirectory

        self.speedRange = speedRange  # どの速度まで推論するか
        self.speedInterval = speedInterval

        self.throttleRange = throttleRange  # どの大きさのthrottleまで推論するか
        self.throttleInterval = throttleInterval

        '''
        推論用のデータを作っておく
        '''
        __speedRef = np.arange(0, self.speedRange, self.speedInterval)
        __throttleRef = np.arange(0, self.throttleRange, self.throttleInterval)

        # meshgridを作成
        self.throttleResult, self.speedResult = np.meshgrid(__throttleRef, __speedRef)

        self.speedRefNum = int(self.speedRange / self.speedInterval)
        self.throttleRefNum = int(self.throttleRange / self.throttleInterval)

        self.__throttleModelGrouping(speedGroupNum, throttleGroupNum)
        self.__findOutliers()
        self.__createTrainingData()

        if train:
            self.__train(save)
        else:
            self.model = model_from_json(open(self.resultDirectory + 'TrainedModel/throttleModel.json').read())
            self.model.load_weights(self.resultDirectory + 'TrainedModel/throttleModel.h5')

        self.__refer()

    def __removeOutliers(self, row):
        # 1を外れ値とする
        stddev = self.groupThrottleStdList[row.groupThrottle]
        mean = self.groupThrottleMeanList[row.groupThrottle]

        if stddev == 0:  # データが1個しかないとき
            return 0

        elif np.abs(row["accFiltered"] - mean) / stddev > 1:
            return 1

        else:
            return 0

    def __throttleModelGrouping(self, speedGroupNum, throttleGroupNum):
        self.dfPacmod = self.dfPacmod.query("0<speed<20 & accel>0.012 & accFiltered>0")

        self.dfPacmod["speedKey"] = pd.cut(self.dfPacmod.speed, speedGroupNum,
                                           labels=[i + 1 for i in range(speedGroupNum)])
        self.dfPacmod["throttleKey"] = pd.cut(self.dfPacmod.accel, throttleGroupNum,
                                              labels=[i + 1 for i in range(throttleGroupNum)])
        _groupListAcc = list(self.dfPacmod.groupby(["speedKey", "throttleKey"], as_index=False).groups.keys())
        self.dfPacmod["groupThrottle"] = self.dfPacmod.apply(lambda row: _groupListAcc.index((row["speedKey"],
                                                                                              row["throttleKey"])),
                                                             axis=1)

        self.groupThrottleMeanList = self.dfPacmod.groupby("groupThrottle").mean()["accFiltered"].values
        self.groupThrottleStdList = self.dfPacmod.groupby("groupThrottle").std(ddof=0)["accFiltered"].values

    def __findOutliers(self):
        self.dfPacmod["remove"] = self.dfPacmod.apply(self.__removeOutliers, axis=1)

    def __createNetwork(self):
        model = Sequential()
        model.add(Dense(64, input_dim=2, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer="Adam", loss="mean_squared_error")
        self.model = model

    def __createTrainingData(self):
        self.dfPacmodTrain = self.dfPacmod.query("remove==0 & speed<20").copy()
        self._accMinNum = self.dfPacmodTrain["accFiltered"].min()
        self._accMaxNum = self.dfPacmodTrain["accFiltered"].max()


    def __train(self, save):

        # dataset
        x_train = np.array([self.dfPacmodTrain["accel"], self.dfPacmodTrain["speed"] / self.dfPacmodTrain["speed"].max()]).T
        y_train = (self.dfPacmodTrain["accFiltered"].values - self._accMinNum) / (self._accMaxNum - self._accMinNum)

        self.__createNetwork()
        self.model.fit(x_train, y_train, batch_size=128, epochs=300, verbose=1, validation_split=0.2, shuffle=True)

        if save:
            json_string = self.model.to_json()
            open(self.resultDirectory + 'TrainedModel/throttleModel.json', 'w').write(json_string)
            self.model.save_weights(self.resultDirectory + 'TrainedModel/throttleModel.h5')

    def __refer(self):
        _predict = []
        for i in range(0, self.speedRefNum):
            _predict.append(self.model.predict(np.array([self.throttleResult[i],
                                                        self.speedResult[i] / self.speedResult[:, 0].max()]).T))

        self.predictedAcceleration = np.array(_predict).reshape(self.speedRefNum, self.throttleRefNum)

    def getGraphResult(self):
        _representSpeed = self.dfPacmod.query("remove==0").groupby("groupThrottle").mean()["speed"].values
        _representThrottle = self.dfPacmod.query("remove==0").groupby("groupThrottle").mean()["accel"].values
        _representAcceleration = self.dfPacmod.query("remove==0").groupby("groupThrottle").mean()["accFiltered"].values

        _surface = Surface(x=self.throttleResult, y=self.speedResult,
                           z=self.predictedAcceleration * (self._accMaxNum - self._accMinNum) + self._accMinNum,
                           colorscale='YlGnBu', name="predictSurface")

        _scatterRepresent = Scatter3d(x=_representThrottle, y=_representSpeed, z=_representAcceleration, mode='markers',
                                      marker=dict(size=2, color="green"), name="Boss")

        _scatterRemove = Scatter3d(x=self.dfPacmod.query("remove==0").accel, y=self.dfPacmod.query("remove==0").speed,
                                   z=self.dfPacmod.query("remove==0").accFiltered, mode='markers',
                                   marker=dict(size=3, color="blue"), name="RemoveOutliers")

        layout = go.Layout(title="ThrottleModel", scene=dict(xaxis=dict(title="CMD(Throttle)", range=[0.1, 0.7]),
                                                             yaxis=dict(title="speed", range=[20, 0]),
                                                             zaxis=dict(title="acceleration", range=[0, 3])))

        fig = go.Figure(data=[_surface, _scatterRepresent, _scatterRemove], layout=layout)

        _figName = "throttle.html"
        print("saveGraphFileName:" + self.resultDirectory + _figName)
        plotly.offline.plot(fig, filename=self.resultDirectory + _figName)

    def getCsvResult(self):
        _accResult = self.predictedAcceleration * (self._accMaxNum - self._accMinNum) + self._accMinNum
        output = pd.DataFrame({'throttle': self.throttleResult.reshape(-1, ),
                               'speed': self.speedResult.reshape(-1, ),
                               'acceleration': _accResult.reshape(-1, )})

        output = output.round({'throttle': 2, 'acceleration': 2, 'speed': 2})

        output.to_csv(self.resultDirectory + "throttle.csv", index=False)
