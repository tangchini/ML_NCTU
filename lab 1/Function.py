
import time
import math
import numpy as np
import copy


def getMean2D(input):
    return np.mean(input, axis=1)
    
def getTotalAxes(xAxes,yAxes,zAxes):
    totalAxes = np.sqrt(xAxes * xAxes + yAxes * yAxes + zAxes * zAxes);
    return totalAxes

def getRoll(Ax,Az):
    roll = np.arctan2(-Ax,Az)*180 / np.pi
    return roll

def getPitch(Ay,Az):
    pitch = -(np.arctan2(-Ay,Az)*180) / np.pi
    return pitch