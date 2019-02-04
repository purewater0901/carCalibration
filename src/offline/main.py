import sys
from src.offline.tool.process import Process
from src.offline.tool.throttle import Throttle
from src.offline.tool.brake import Brake

'''
第1引数にpacmodのpath(defalut: ./data/pacmodLongTokyoDistrict.csv)
第2引数にimuのpath(defalut: ./data/imuLongTokyoDistrict.csv)
第3引数に出力のディレクトリ(default: ./reslut)
'''

if len(sys.argv) > 4:
    pacmodFilePath = sys.argv[1]
    imuFilePath = sys.argv[2]
    outputFilePath = sys.argv[3]

else:
    pacmodFilePath = '/home/yutaka/carCalibration/data/pacmodLongTokyoDistrict.csv'
    imuFilePath    = '/home/yutaka/carCalibration/data/imuLongTokyoDistrict.csv'
    outputFilePath = '/home/yutaka/carCalibration/result/'

data = Process(imuFilePath, pacmodFilePath).getData()

t = Throttle(data, outputFilePath)
t.getGraphResult()
t.getCsvResult()

b = Brake(data, outputFilePath)
b.getGraphResult()
b.getCsvResult()

