# carCalibration

## Table of Contents
1. Offline Table
  - Throttle Model
  - Brake Model
  
2. Online Model
  - Throttle Model
  - Brake Model
  
## Description
_carCalibration_ will show you how to build a data-driven longitudinal control system by machine-learning. 
  
## Dependency
- numpy >= 1.15.4
- pandas >= 0.22.0
- scipy >= 1.2.0
- tensorflow >= 1.7.0
- keras >= 2.1.6
- python3

###### if you wish to use matplotlib or plotly then you should use
- matplotlib >= 2.2.2
- plotly >= 2.5.1

## Usage

### Input Csv
1. From Pacmod 
You need accel, brake, speed, steer, leftWheelSpeed, rightWheelSpeed

| %time | accel | brake | speed | steer | leftWheelSpeed | rightWheelSpeed | 
----- | ----- | ----- | ----- | ----- | -----| ----- |
| 0 | 0 | 0.4 | 0 | 0.2 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... |

2. From Imu
You need x, y, z direction acceleration and pitch angle

| %time | x | y | z | pitch |
----- | ----- | ----- | ----- | ----- |
| 0 | 0 | 0 | -9.8 | 0 |
| .... | .... | .... | .... | .... |

- Prepare your csv File
- copy csv file under the data directory
- run the following command
`python src/main.py`

### Output Csv
- Result will output in the 'result/' directory 
- Output csv file will be below. You will two type csv. (Brake and Throttle)

| command(Throttle or Brake) | speed | accceleration |
------------------- | ------------------- | ------------------- | 
| 0.0 | 0 | 0.0 |
| .... | .... | .... |

## References
https://arxiv.org/abs/1808.10134
