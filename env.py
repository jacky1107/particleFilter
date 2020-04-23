import cv2
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def randomMove(robot):
    x, y, theta = robot
    bound = 30
    step = abs(np.random.normal(0, 20, 1))
    if random.random() >= 0.5: # turn
        if random.random() >= 0.5:
            theta += 40
        else:
            theta -= 40
    else: # straight
        x += step * cos(theta)
        y += step * sin(theta)
        if x <= bound:
            x = bound
        elif y <= bound:
            y = bound
        elif x >= 890:
            x = 870
        elif y >= 290:
            y = 270
    return x, y, theta

def findSensorInRange(robot, obstacles, plot=True):
    dists = []
    diffs = []
    robot_x, robot_y, robot_angle = robot
    obstacles_x, obstacles_y, obstacles_angle = obstacles[:,0], obstacles[:,1], obstacles[:,2]
    angles = tan(obstacles_y - robot_y, obstacles_x - robot_x)
    for index, angle in enumerate(angles):
        diff = robot_angle - angle
        if diff >= 180: diff = diff - 360
        if diff >= -40 and diff <= 40:
            dist = cal_distance(robot_x, robot_y, obstacles[index,0], obstacles[index,1])
            dists.append(dist)
            diffs.append(diff)
    return dists, diffs

def resampling(particles, weights):
    sigma = copy.copy(weights[::-1])
    weights = np.cumsum(weights[::-1])
    weights = weights / max(weights)
    weights = weights[::-1]
    temp = weights[0]
    for i in range(1, len(weights)):
        weights[i-1] = temp - weights[i]
        temp = weights[i]

    index = []
    for i in range(len(weights)):
        index.append(np.random.choice(np.arange(len(weights)), p=weights))

    i = 0
    for j, count in enumerate(np.bincount(index)):
        for _ in range(count):
            (x, y, t) = particles[j]
            particles[i, 0] = np.random.normal(x, sigma[i])
            particles[i, 1] = np.random.normal(y, sigma[i])
            particles[i, 2] = np.random.normal(t, sigma[i])
            i += 1
    return particles


def sort(particles, weights):
    return particles[np.argsort(weights)[::-1]], sorted(weights)[::-1]

def cal_distance(x1, y1, x2, y2):
    return np.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )

def cdf(x, mean, sigma):
    pb = 0.5*(1+erf((x-mean)/(sigma*np.sqrt(2))))
    return (1 - pb) / 0.5 if pb > 0.5 else pb / 0.5

def normal(x, mean, sigma):
    return 1/(sigma*np.sqrt(2*(np.pi)))*np.exp(-(x-mean)**2/2*sigma**2)

def sin(theta):
    return np.sin(theta*np.pi/180)

def cos(theta):
    return np.cos(theta*np.pi/180)

def tan(y, x):
    return np.arctan2(y, x) * 180 / np.pi
