import cv2
import time
import numpy as np
import scipy.stats as st

from config import *

obstaclesNumber = 10
particlesNumber = 500

height = 300
width = 900
channels = 3

env = np.zeros((height, width, channels))
env.fill(255)

obstacles, robotInfo = generateObstaclesAndRobot(env, obstaclesNumber=obstaclesNumber)
particles = generateParticles(env, particlesNumber=particlesNumber)
weights = np.ones(particlesNumber)

for time in range(100):
    # Visualization
    env.fill(255)
    drawObstaclesOrParticles(env, obstacles, (0, 0, 0), wall=True)
    drawObstaclesOrParticles(env, particles, (0, 255, 0))
    drawRobot(env, robotInfo)

    # Calculate Control
    robotInfo2 = moveRobot(env, robotInfo)
    dist = np.linalg.norm(robotInfo[:2] - robotInfo2[:2])
    head = robotInfo2[-1]
    u = np.array([head, dist])
    robotInfo = robotInfo2

    # Use Control Input to predict particles
    particles = predict(env, particles, u)

    measurement = np.linalg.norm(obstacles[:, :2] - robotInfo[:2], axis=1)

    weights = update(particles, weights, measurement, obstacles)
    indexes = systematicResample(weights)
    particles, weights = resampleFromIndex(particles, weights, indexes)

    mean_x = int(np.mean(particles[:, 0]))
    mean_y = int(np.mean(particles[:, 1]))

    x, y, _ = robotInfo
    loss = np.sqrt((mean_x - x) ** 2 + (mean_y - y) ** 2)
    cv2.circle(env, (mean_x, mean_y), 3, (255, 0, 0), -1)

    cv2.imshow("env", env)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

print(f'loss: {loss}')
cv2.waitKey(0)
cv2.destroyAllWindows()