# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

class Vehicle():
    def __init__(self, init_x, init_y, v_x, v_y, theta, length, width, dt):
        self.x = init_x
        self.y = init_y
        self.v_x = v_x
        self.v_y = v_y
        self.theta = theta
        self.length = length
        self.width = width
        self.dt = dt

    def state_update(self, V, YR):
        #座標系に注意
        self.x += V*np.sin(self.theta) * self.dt # longitudinal coordinate value
        self.y += V*np.cos(self.theta) * self.dt # lateral coordinate value
        self.theta += YR * self.dt

    def step(self, V, YR):
        self.state_update(V, YR)
        state = np.array([self.x, self.y, self.v_x, self.v_y, self.theta])
        # state = np.array([[self.x],
        #                 [self.y],
        #                 [self.v_x],
        #                 [self.v_y],
        #                 [self.theta]])
        # state = np.array([self.y])
        state = np.reshape(state, [1, 5])
        reward = 0

        if abs(self.x) > 1:
            reward = -1
            done = True
        elif abs(self.y) > 50:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        return state, reward, done


class Animation():
    def __init__(self):
        ## plot 初期化
        # グラフ仕様設定
        self.fig = plt.figure(figsize=(3,15))

        # 軸
        # 最大値と最小値⇒軸の範囲設定
        self.max_x = 5
        self.min_x = -5
        self.max_y = 100
        self.min_y = 0


    def plot_rectangle(self, Car0):
        # Axes インスタンスを作成
        ax = self.fig.add_subplot(111)

        ax.set_xlim(self.min_x, self.max_x)
        ax.set_ylim(self.min_y, self.max_y)

        # # 軸の縦横比, 正方形，単位あたりの長さを等しくする
        # ax.set_aspect('equal')
        self.change_aspect_ratio(ax, 1/5) # 横を1/5倍長く（縦を5倍長く）設定

        # 軸の名前設定
        ax.set_xlabel('Y [m]')
        ax.set_ylabel('X [m]')

        # その他グラフ仕様
        ax.grid(True) # グリッド
        # 凡例
        # ax.legend()

        # rectangle
        rect_0 = plt.Rectangle((Car0.x-Car0.width/2, Car0.y-Car0.length/2),Car0.width,Car0.length,angle=Car0.theta,fc="#770000")
        ax.add_patch(rect_0)
        plt.pause(.05)

        self.fig.delaxes(ax)


    def change_aspect_ratio(self, ax, ratio):
        '''
        This function change aspect ratio of figure.
        Parameters:
            ax: ax (matplotlit.pyplot.subplots())
                Axes object
            ratio: float or int
                relative x axis width compared to y axis width.
        '''
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect)

    def close_figure(self):
        plt.cla()
        plt.clf()
        plt.close()
