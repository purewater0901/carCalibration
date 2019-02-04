import sys
import numpy as np
import pandas as pd
from scipy import signal


class Process:

    def __init__(self, imuFileName, pacmodFileName, speedMeanFilNum=5, accMeanFilNum=20):
        if imuFileName == '' or pacmodFileName == '':
            print("ファイル名が指定されていません")
            sys.exit(1)

        self.imuFileName = imuFileName
        self.pacmodFileName = pacmodFileName

        self.__readFile()
        self.__preprocess(speedMeanFilNum, accMeanFilNum)


    def __readFile(self):
        self.dfImu = pd.read_csv(self.imuFileName)
        self.dfPacmod = pd.read_csv(self.pacmodFileName)

    def __getNearestValue(self, row):
        idx = np.abs(self.timeImuList - row).argmin()
        return pd.Series([self.accelerationList[idx], self.imuPitchList[idx]])


    def __preprocess(self, speedMeanFilNum, accMeanFilNum):
        self.dfImu = self.dfImu.rename(columns={'%time': 'time'})
        self.dfPacmod = self.dfPacmod.rename(columns={'%time': 'time'})
        self.dfImu["time"] = self.dfImu["time"] - self.dfImu["time"][0]
        self.dfPacmod["time"] = self.dfPacmod["time"] - self.dfPacmod["time"][0]
        self.dfImu["time"] = self.dfImu["time"] / (10 ** 9)
        self.dfPacmod["time"] = self.dfPacmod["time"] / (10 ** 9)

        self.dfImu["time"] = self.dfImu["time"] - 0.4  # 入力遅れの分の加算

        self.dfImu = self.dfImu.dropna()
        self.dfPacmod = self.dfPacmod.dropna()

        self.dfPacmod = self.dfPacmod[self.dfPacmod.speed > 0]  # 速度が0のところはとりあえず無視
        self.dfPacmod["speed"] = (self.dfPacmod["leftWheelSpeed"]
                                  + self.dfPacmod["rightWheelSpeed"]) / 2  # 車速にwheelSpeedを用いる
        self.dfPacmod["speed"] = self.dfPacmod["speed"]/3.6
        self.dfImu["x"] = -1 * self.dfImu["x"]

        # 速度の移動平均を求める
        _speedFil = np.ones(speedMeanFilNum) / speedMeanFilNum
        self.dfPacmod["speed"] = np.convolve(self.dfPacmod["speed"], _speedFil, mode='same')  # 移動平均

        # 速度の微分から加速度を求める
        _deltaT = np.diff(self.dfPacmod["time"]).mean()
        self.dfPacmod["accFromSpeed"] = np.insert(np.diff(self.dfPacmod["speed"]) / _deltaT, 0, 0)

        # 加速度にフィルターをかける
        _accelerationFil = np.ones(accMeanFilNum) / accMeanFilNum
        self.dfPacmod["accFiltered"] = np.convolve(self.dfPacmod["accFromSpeed"], _accelerationFil,
                                                   mode='same')  # 移動平均

        # 今回は使用しないがImuから得られた加速度にフィルターをかける
        _b, _a = signal.butter(6, 0.1, 'low')  # 次数6でカットオフ周波数は5Hz
        self.dfImu["acceleration"] = signal.filtfilt(_b, _a, self.dfImu["x"])

        self.accelerationList = self.dfImu["acceleration"].values
        self.imuPitchList = self.dfImu["pitch"].values
        self.timeImuList = self.dfImu["time"].values

        # ImuのデータとPacmodのデータを合わせる
        self.dfPacmod[["acceleration", "pitch"]] = self.dfPacmod.time.apply(self.__getNearestValue)

        self.dfPacmod = self.dfPacmod[abs(self.dfPacmod.steer) < 0.5]  # steer角が大きいところは除去
        self.dfPacmod = self.dfPacmod[np.abs(self.dfPacmod.pitch - self.dfPacmod.pitch.mean()) /
                                      self.dfPacmod.pitch.std() <= 1]

    def getData(self, speedGroupNum=10, throttleGroupNum=10):
        return self.dfPacmod
