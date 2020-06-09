from scipy.special import erf
import numpy as np
from env import *

class Particle_filter():
    def __init__(self, PARTICLE_NUMBER, OBSTACLE_NUMBER):
        # ROBOT
        self.HEIGHT = 300
        self.WIDTH  = 900
        self.ROBOT_SIZE = 5

        # OBSTACLE
        self.OBSTACLE_NUMBER = OBSTACLE_NUMBER
        self.OBSTACLE_LENTH = 10
        self.OBSTACLE_SIZE = 5
        self.obstacles = np.random.random((self.OBSTACLE_NUMBER, 3)) * (self.WIDTH, self.HEIGHT, 360)
        
        # PRATICLE
        self.PARTICLE_NUMBER = PARTICLE_NUMBER
        self.PARTICLE_SIZE = 5
        self.particles = np.random.random((self.PARTICLE_NUMBER, 3)) * (self.WIDTH, self.HEIGHT, np.random.uniform(0, 90))

        # RECREATED
        self.dropout = 0.1

    def robot_init(self):
        return (self.WIDTH/2, self.HEIGHT/2, 90)

    def generate_obstacles(self):
        x1 = self.obstacles[:,0] - self.OBSTACLE_LENTH * abs(cos(self.obstacles[:,2]))
        x2 = self.obstacles[:,0] + self.OBSTACLE_LENTH * abs(cos(self.obstacles[:,2]))
        y1 = self.obstacles[:,1] - self.OBSTACLE_LENTH * abs(sin(self.obstacles[:,2]))
        y2 = self.obstacles[:,1] + self.OBSTACLE_LENTH * abs(sin(self.obstacles[:,2]))
        return x1, x2, y1, y2

    def recreated(self):
        lenth = int(self.PARTICLE_NUMBER * self.dropout)
        self.particles[-lenth:] = np.random.random((lenth, 3)) * (self.WIDTH, self.HEIGHT, np.random.uniform(0, 90))
        return self.particles

    def cal_likelihood(self, robot, dists, diffs, weights):
        real_dists, real_diffs= findSensorInRange(robot, self.obstacles)
        real_dists = sorted(real_dists)
        real_lenth = len(real_dists)

        predict_dists = sorted(dists)
        predict_lenth = len(dists)

        for i, dist in enumerate(real_dists):
            if dist >= 200:
                pAB0 = 0.1
                pAB1 = 0.1
                pAB2 = 0.05
                pAB3 = 0.3
            elif dist < 50:
                pAB0 = 0.7
                pAB1 = 0.05
                pAB2 = 0.1
                pAB3 = 10
            else:
                pAB0 = 0.8
                pAB1 = 0.01
                pAB2 = 0.01
                pAB3 = 0.1
            sigma1 = pAB1 * dist
            sigma2 = pAB2 * dist
            sigma3 = pAB3 * dist
            if i >= predict_lenth:
                PP = 1 - pAB0
                weights[i] = float(weights[i] * PP)
            else:
                t = np.random.normal(self.obstacles[i,2], sigma2)
                PP1 = cdf(predict_dists[i], dist, sigma1)
                PP2 = cdf(t, self.obstacles[i,2], sigma2)
                PP3 = cdf(diffs[i], real_diffs[i], sigma3)
                weights[i] = float(weights[i] * PP1 * PP2 * PP3)
        if real_lenth < predict_lenth:
            for i in range(real_lenth, predict_lenth, 1):
                if predict_dists[i] >= 200:
                    weights[i] = float(weights[i] * (1 - 0.95))
                elif predict_dists[i] < 50:
                    weights[i] = float(weights[i] * (1 - 0.9))
                else:
                    weights[i] = float(weights[i] * (1 - 0.99))
        return weights
            
