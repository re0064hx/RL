# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

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
        state = np.reshape(state, [1, 5])

        if abs(self.x) > 0.5:
            reward = -1
            done = True
        elif abs(self.y) > 30:
            reward = 2
            done = True
        else:
            reward = 1
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


    def plot_rectangle(self, Car0, Car1, Car2, Car3, Car4):
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
        rect_0 = patches.Rectangle((Car0.x-Car0.width/2, Car0.y-Car0.length/2),Car0.width,Car0.length,angle=Car0.theta,ec='r', fill=False)
        rect_1 = patches.Rectangle((Car1.x-Car1.width/2, Car1.y-Car1.length/2),Car1.width,Car1.length,angle=Car1.theta,ec='b', fill=False)
        rect_2 = patches.Rectangle((Car2.x-Car2.width/2, Car2.y-Car2.length/2),Car2.width,Car2.length,angle=Car2.theta,ec='b', fill=False)
        rect_3 = patches.Rectangle((Car3.x-Car3.width/2, Car3.y-Car3.length/2),Car3.width,Car3.length,angle=Car3.theta,ec='b', fill=False)
        rect_4 = patches.Rectangle((Car4.x-Car4.width/2, Car4.y-Car4.length/2),Car4.width,Car4.length,angle=Car4.theta,ec='b', fill=False)
        ax.add_patch(rect_0)
        ax.add_patch(rect_1)
        ax.add_patch(rect_2)
        ax.add_patch(rect_3)
        ax.add_patch(rect_4)
        plt.pause(.00001)

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
        # plt.cla()
        # plt.clf()
        plt.close(self.fig)
