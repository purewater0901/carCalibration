import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
import plotly.graph_objs as go
import plotly
from plotly.graph_objs import Surface
from plotly.graph_objs import Scatter3d


class Brake:

    def __init__(self, data, resultDirectory, speedGroupNum=60, brakeGroupNum=25, speedRange=23, brakeRange=1.0,
                 speedInterval=0.1, brakeInterval=0.01, train=True, save=True):
        self.dfPacmod = data.copy()
        self.resultDirectory = resultDirectory

        self.speedRange = speedRange  # どの速度まで推論するか
        self.speedInterval = speedInterval

        self.brakeRange = brakeRange  # どの大きさのbrakeまで推論するか
        self.brakeInterval = brakeInterval

        '''
        推論用のデータを作っておく
        '''
        __speedRef = np.arange(0, self.speedRange, self.speedInterval)
        __brakeRef = np.arange(0, self.brakeRange, self.brakeInterval)

        # meshgriを作成
        self.brakeResult, self.speedResult = np.meshgrid(__brakeRef, __speedRef)

        self.speedRefNum = int(self.speedRange / self.speedInterval)
        self.brakeRefNum = int(self.brakeRange / self.brakeInterval)

        self.__brakeModelGrouping(speedGroupNum, brakeGroupNum)
        self.__findOutliers()
        self.__createTrainingData()

        if train:
            self.__train(save)
        else:
            self.model = model_from_json(open(self.resultDirectory + 'TrainedModel/brakeModel.json').read())
            self.model.load_weights(self.resultDirectory + 'TrainedModel/brakeModel.h5')

        self.__refer()

    def __removeOutliers(self, row):
        # 1を外れ値とする
        stddev = self.groupBrakeStdList[row.groupBrake]
        mean = self.groupBrakeMeanList[row.groupBrake]

        if stddev == 0:  # データが1個しかないとき
            return 0

        elif np.abs(row["accFiltered"] - mean) / stddev > 1:
            return 1

        else:
            return 0

    def __brakeModelGrouping(self, speedGroupNum, brakeGroupNum):
        self.dfPacmod = self.dfPacmod.query("0<speed<23 & brake>0.025 & accFiltered<-0.18")

        self.dfPacmod["speedKey"] = pd.cut(self.dfPacmod.speed, speedGroupNum,
                                           labels=[i + 1 for i in range(speedGroupNum)])
        self.dfPacmod["brakeKey"] = pd.cut(self.dfPacmod.brake, brakeGroupNum,
                                           labels=[i + 1 for i in range(brakeGroupNum)])
        _groupListAcc = list(self.dfPacmod.groupby(["speedKey", "brakeKey"], as_index=False).groups.keys())
        self.dfPacmod["groupBrake"] = self.dfPacmod.apply(lambda row: _groupListAcc.index((row["speedKey"],
                                                                                           row["brakeKey"])), axis=1)

        self.groupBrakeMeanList = self.dfPacmod.groupby("groupBrake").mean()["accFiltered"].values
        self.groupBrakeStdList = self.dfPacmod.groupby("groupBrake").std(ddof=0)["accFiltered"].values

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
        self.dfPacmodTrain = self.dfPacmod.query("remove==0 & speed>2").copy()
        self._accMinNum = self.dfPacmodTrain["accFiltered"].min()
        self._accMaxNum = self.dfPacmodTrain["accFiltered"].max()

    def __train(self, save):
        x_train = np.array([self.dfPacmodTrain["brake"], self.dfPacmodTrain["speed"] / self.dfPacmodTrain["speed"].max()]).T
        y_train = (self.dfPacmodTrain["accFiltered"].values - self._accMinNum) / (self._accMaxNum - self._accMinNum)

        self.__createNetwork()
        self.model.fit(x_train, y_train, batch_size=128, epochs=300, verbose=1, validation_split=0.2, shuffle=True)

        if save:
            json_string = self.model.to_json()
            open(self.resultDirectory + 'TrainedModel/brakeModel.json', 'w').write(json_string)
            self.model.save_weights(self.resultDirectory + 'TrainedModel/brakeModel.h5')

    def __refer(self):
        _predict = []
        for i in range(0, self.speedRefNum):
            _predict.append(self.model.predict(np.array([self.brakeResult[i],
                                                         self.speedResult[i] / self.speedResult[:, 0].max()]).T))

        self.predictedAcceleration = np.array(_predict).reshape(self.speedRefNum, self.brakeRefNum)

    def getGraphResult(self):
        _representSpeed = self.dfPacmod.query("remove==0").groupby("groupBrake").mean()["speed"].values
        _representBrake = self.dfPacmod.query("remove==0").groupby("groupBrake").mean()["brake"].values
        _representAcceleration = self.dfPacmod.query("remove==0").groupby("groupBrake").mean()["accFiltered"].values

        _surface = Surface(x=self.brakeResult, y=self.speedResult,
                           z=self.predictedAcceleration * (self._accMaxNum - self._accMinNum) + self._accMinNum,
                           colorscale='YlGnBu', name="predictSurface")

        _scatterRepresent = Scatter3d(x=_representBrake, y=_representSpeed, z=_representAcceleration, mode='markers',
                                      marker=dict(size=2, color="green"), name="Boss")

        _scatterRemove = Scatter3d(x=self.dfPacmod.query("remove==0").brake, y=self.dfPacmod.query("remove==0").speed,
                                   z=self.dfPacmod.query("remove==0").accFiltered, mode='markers',
                                   marker=dict(size=3, color="blue"), name="RemoveOutliers")

        layout = go.Layout(title="ThrottleModel", scene=dict(xaxis=dict(title="CMD(Brake)", range=[0, 0.7]),
                                                             yaxis=dict(title="speed", range=[20, 0]),
                                                             zaxis=dict(title="acceleration", range=[-3, 0])))

        fig = go.Figure(data=[_surface, _scatterRepresent, _scatterRemove], layout=layout)

        _figName = "brake.html"
        print("saveGraphFileName:" + self.resultDirectory + _figName)
        plotly.offline.plot(fig, filename=self.resultDirectory + _figName)

    def getCsvResult(self):
        _accResult = self.predictedAcceleration * (self._accMaxNum - self._accMinNum) + self._accMinNum
        output = pd.DataFrame({'brake': self.brakeResult.reshape(-1, ),
                               'speed': self.speedResult.reshape(-1, ),
                               'acceleration': _accResult.reshape(-1, )})

        output = output.round({'brake': 2, 'acceleration': 2, 'speed': 2})

        output.to_csv(self.resultDirectory + "brake.csv", index=False)
