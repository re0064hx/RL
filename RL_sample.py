# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # gymの画像保存
from keras import backend as K
import tensorflow as tf
import simulator as env

NUM_EPISODES = 1000


def main():
    # Set sampling time
    dt = 0.05
    MaxRange = 100

    # Deep Q-learning
    '''
    State   :   x, y, vx, vy, theta
    Action  :   V, YR
    '''

    for episode in range(NUM_EPISODES):
        terminal = False
        i = 0
        #車両インスタンス作成(車両座標系)
        # Initialization : (x, y, vx, vy, theta, length, width, dt)
        Car0 = env.Vehicle(0, 0, 10, 0, 0, 4.75, 1.75, dt)
        Car1 = env.Vehicle(30, 1.75, 9, 0, 0, 4.75, 1.75, dt)
        Car2 = env.Vehicle(-30, 1.75, 9, 0, 0, 4.75, 1.75, dt)
        Car3 = env.Vehicle(15, -1.75, 9, 0, 0, 4.75, 1.75, dt)
        Car4 = env.Vehicle(-15, -1.75, 9, 0, 0, 4.75, 1.75, dt)
        #描画インスタンス作成
        drawer = env.Animation()

        while not terminal:
            i += 1
            V = 10
            if i<=100:
                YR = np.sin(2*np.pi/100*i)/10
            else:
                YR = 0
            Car0.state_update(V, YR)
            # drawer.plot_rectangle(Car0)
            print('\r Episode:%5d, LoopTime:%4d' % (episode, i), end='')

            if i == MaxRange:
                terminal = True
                drawer.close_figure()

        if i == NUM_EPISODES:
            drawer.close_figure()
            print('')

    # env.sim(Car0, drawer)


if __name__ == '__main__':
    main()
