import cv2
import time
import random
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from env import *
from particle_class import Particle_filter
OBSTACLE_NUMBER = 40
PARTICLE_NUMBER = 1000

P = Particle_filter(PARTICLE_NUMBER=PARTICLE_NUMBER, OBSTACLE_NUMBER=OBSTACLE_NUMBER)
x1, x2, y1, y2 = P.generate_obstacles()
robot = P.robot_init()
weights = np.ones(P.PARTICLE_NUMBER)

# init
for i in range(30):
    plt.xlim(0,P.WIDTH)
    plt.ylim(0,P.HEIGHT)
    robot = randomMove( robot )
    for j, particle in enumerate(P.particles):
        dists, diffs = findSensorInRange(particle, P.obstacles, plot=False)
        weights = P.cal_likelihood(robot, dists, diffs, weights)
    P.particles, weights = sort(P.particles, weights)
    
    top10 = P.particles[:10]
    # draw
    plt.scatter(robot[0], robot[1], c='r',s=P.ROBOT_SIZE)
    plt.scatter(P.particles[:,0],P.particles[:,1],c='b',s=P.PARTICLE_SIZE)
    plt.scatter(top10[:,0],top10[:,1],c='r',s=P.PARTICLE_SIZE)
    plt.scatter(P.obstacles[:,0],P.obstacles[:,1],c='k',s=P.OBSTACLE_SIZE)
    plt.plot([x1, x2], [y1, y2] ,c='k')
    plt.plot([robot[0],robot[0]+10*cos(robot[2])],[robot[1],robot[1]+10*sin(robot[2])],c='r')
    plt.pause(0.0001)
    plt.clf()
    P.particles = resampling(P.particles, weights)
    P.particles = P.recreated()
    print(i)
plt.show()