import cv2
import numpy as np
import scipy.stats as st

def addNoiseOnMeasurement(measurement):
    prob = []
    angleStd = []
    distanceStd = []
    for i, measure in enumerate(measurement):
        if measure >= 200:
            prob.append(0.1)
            angleStd.append(0.05 * measure)
            distanceStd.append(0.1 * measure)
        elif measure >= 50:
            prob.append(0.8)
            angleStd.append(0.01 * measure)
            distanceStd.append(0.01 * measure)
        else:
            prob.append(0.7)
            angleStd.append(0.1 * measure)
            distanceStd.append(0.05 * measure)
    return np.array(distanceStd), np.array(angleStd), np.array(prob)

def predict(env, particles, u, dt=1.):
    h, w, c = env.shape
    dist = (u[1] * dt)
    particles[:, 0] += cos(u[0]) * dist
    particles[:, 1] += sin(u[0]) * dist
    particles[:, 0] = np.clip(particles[:, 0], 0, w)
    particles[:, 1] = np.clip(particles[:, 1], 0, h)
    return particles

def neff(weights):
    return 1 / np.sum(weights ** 2)

def update(particles, weights, measurement, obstacles):
    for i, obstacle in enumerate(obstacles):
        angles = particles[:, -1]
        distance = np.power((particles[:,0] - obstacle[0]) ** 2 + (particles[:,1] - obstacle[1]) ** 2, 0.5)
        distanceStd, angleStd, prob = addNoiseOnMeasurement(distance)
        pdf = st.norm(distance, distanceStd).pdf(measurement[i])
        pdf = pdf / np.sum(pdf)
        weights = weights * prob * pdf
    weights += 1.e-300
    weights = weights / np.sum(weights)
    return weights

def systematicResample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resampleFromIndex(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights = weights / np.sum(weights)
    particles[:,0] = particles[:,0] + np.random.normal(0, 5, len(particles))
    particles[:,1] = particles[:,1] + np.random.normal(0, 5, len(particles))
    particles[:,2] = particles[:,2] + np.random.normal(0, 5, len(particles))
    return particles, weights

def generateParticles(env, particlesNumber=100):
    h, w, c = env.shape
    particles = []
    for i in range(particlesNumber):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        orientation = np.random.randint(0, 360)
        particles.append([x, y, orientation])
    return np.array(particles, dtype="float64")

def generateObstaclesAndRobot(env, obstaclesNumber=10):
    h, w, c = env.shape
    obstacles = []
    for i in range(obstaclesNumber):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        orientation = np.random.randint(0, 360)
        obstacles.append([x, y, orientation])

    x = np.random.randint(0, w)
    y = np.random.randint(0, h)
    orientation = 0
    robotInfo = [x, y, orientation]
    return np.array(obstacles), np.array(robotInfo)

def drawObstaclesOrParticles(env, obstacles, color, wall=False):
    h, w, c = env.shape
    wallLength = 4
    for i in range(len(obstacles)):
        x, y, orientation = obstacles[i]
        x1 = int( (x - wallLength) )
        y1 = int( (y - wallLength) )
        x2 = int( (x + wallLength) )
        y2 = int( (y + wallLength) )
        if wall: cv2.rectangle(env, (x1, y1), (x2, y2), color, -1)
        else: cv2.circle(env, (int(x), int(y)), 2, color, -1)

def drawRobot(env, robotInfo):
    h, w, c = env.shape
    x, y, orientation = robotInfo
    cv2.circle(env, (int(x), int(y)), 2, (0,0,255), -1)

def moveRobot(env, robotInfo):
    x, y, orientation = robotInfo
    if np.random.randint(2) == 0:
        theta = np.random.normal(45, 0.01 * 45)
        orientation = orientation + theta
        
        if orientation >= 360: orientation = orientation - 360
        elif orientation <= 0: orientation = 0 - orientation
        robotInfo = [x, y, orientation]
    else:
        x2, y2 = drawRotation(env, robotInfo)
        robotInfo = [x2, y2, orientation]
    return np.array(robotInfo)

def drawRotation(env, robotInfo):
    h, w, c = env.shape
    length = np.random.normal(10, 0.1 * 1.5)
    x, y, orientation = robotInfo
    x2 = int(x + length * cos(orientation))
    y2 = int(y + length * sin(orientation))
    x2 = np.clip(x2, 0, w)
    y2 = np.clip(y2, 0, h)
    cv2.line(env, (int(x), int(y)), (int(x2), int(y2)), (0,0,255), 1)
    return x2, y2

def sin(theta):
    return np.sin(theta * np.pi / 180)

def cos(theta):
    return np.cos(theta * np.pi / 180)

def calcAngles(particles, obstacle):
    pass
