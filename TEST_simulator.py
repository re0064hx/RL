# -*- coding: utf-8 -*-
import time
import math
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
        # 座標系に注意
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
    def __init__(self, state_history, num_loop):
        ## plot 初期化
        # グラフ仕様設定
        self.fig = plt.figure(figsize=(3,15))

        # 軸
        # 最大値と最小値⇒軸の範囲設定
        self.max_x = 5
        self.min_x = -5
        self.max_y = 100
        self.min_y = 0

        # Axesインスタンスを作成
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

        # 学習時の状態データ
        self.state_x = state_history[:,0]
        self.state_y = state_history[:,1]

        # Initialize image data
        self.color = []
        self.vehicle_img, = ax.plot([], [], color="b")

        self.num_loop = num_loop

    def plot_rectangle(self, center_x, center_y):
        # 初期化
        self.vehicle_x = [] # 位置を表す円のx
        self.vehicle_y = [] # 位置を表す円のy
        circle_size = 0.5
        Width = 1.7
        Length = 4.7
        # steps = 10 # 円を書く分解能はこの程度で大丈夫
        #
        # for i in range(steps):
        #     self.vehicle_x.append(center_x + circle_size*math.cos(i*2*math.pi/steps))
        #     self.vehicle_y.append(center_y + circle_size*math.sin(i*2*math.pi/steps))
        self.vehicle_x = [center_x-Width/2, center_x+Width/2, center_x+Width/2, center_x-Width/2, center_x-Width/2]
        self.vehicle_y = [center_y-Length/2, center_y-Length/2, center_y+Length/2, center_y+Length/2, center_y-Length/2]

        self.vehicle_img.set_data(self.vehicle_x, self.vehicle_y) # ここでエラー
        return self.vehicle_img,

    def update_anim(self, i):
        if i>=self.num_loop:
            self.close_figure()

        X = self.state_x[i]
        Y = self.state_y[i]
        # print(X, Y)
        vehicle_img, = self.plot_rectangle(X, Y)
        return vehicle_img,

    def change_aspect_ratio(self, ax, ratio):
        aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.set_aspect(aspect)

    def close_figure(self):
        plt.close(self.fig)
