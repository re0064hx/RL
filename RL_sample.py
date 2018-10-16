# -*- coding: utf-8 -*-
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque
from gym import wrappers  # gymの画像保存
from keras import backend as K
import tensorflow as tf
import simulator as env

NUM_EPISODES = 1000
MAXSTEP = 100
DQN_MODE = 1    # 1がDQN、0がDDQN
LENDER_MODE = 1 # 0は学習後も描画なし、1は学習終了後に描画する
GOAL_AVERAGE_REWARD = 195  # この報酬を超えると学習終了
NUM_CONSECUTIVE_ITERATIONS = 10  # 学習完了評価の平均計算を行う試行回数
TOTAL_REWARD_VEC = np.zeros(NUM_CONSECUTIVE_ITERATIONS)  # 各試行の報酬を格納
GAMMA = 0.99    # 割引係数
ISLEARNED = 0  # 学習が終わったフラグ
ISRENDER = 0  # 描画フラグ
# ---
HIDDEN_SIZE = 16               # Q-networkの隠れ層のニューロンの数
LEARNING_RATE = 0.00001         # Q-networkの学習係数
MEMORY_SIZE = 10000            # バッファーメモリの大きさ
BATCH_SIZE = 32                # Q-networkを更新するバッチの大記載

# Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
    def __init__(self, LEARNING_RATE=0.01, state_size=5, action_size=2, HIDDEN_SIZE=10):
        self.model = Sequential()
        self.model.add(Dense(HIDDEN_SIZE, input_dim = state_size, activation='relu'))
        self.model.add(Dense(HIDDEN_SIZE, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=LEARNING_RATE)  # 誤差を減らす学習方法はAdam
        self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)

    # 重みの学習
    def replay(self, memory, BATCH_SIZE, GAMMA, targetQN):
        inputs = np.zeros((BATCH_SIZE, 5))
        targets = np.zeros((BATCH_SIZE, 2))
        mini_batch = memory.sample(BATCH_SIZE)

        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i+1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値関数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + GAMMA * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            # print(action_b)
            targets[i][action_b] = target               # 教師信号
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定

# Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, BATCH_SIZE):
        idx = np.random.choice(np.arange(len(self.buffer)), size=BATCH_SIZE, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

# カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, targetQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.001 + 0.9 / (1.0+episode)

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = targetQN.model.predict(state)[0]  # エラー発生箇所
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する

        else:
            action = np.random.choice([0, 1])  # ランダムに行動する

        return action

# 損失関数の定義
# 損失関数にhuber関数を使用します 参考https://github.com/jaara/AI-blog/blob/master/CartPole-DQN.py
def huberloss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < 1.0
    L2 = 0.5 * K.square(err)
    L1 = (K.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

def main():
    # Set sampling time
    dt = 0.05
    '''
    === Deep Q-learning ===
    State   :   x, y, vx, vy, theta
    Action  :   V, YR
    '''
    # Qネットワークとメモリ、Actorの生成--------------------------------------------------------
    mainQN = QNetwork(HIDDEN_SIZE=HIDDEN_SIZE, LEARNING_RATE=LEARNING_RATE)     # メインのQネットワーク
    targetQN = QNetwork(HIDDEN_SIZE=HIDDEN_SIZE, LEARNING_RATE=LEARNING_RATE)   # 価値を計算するQネットワーク
    plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
    memory = Memory(max_size=MEMORY_SIZE)
    actor = Actor()

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

        state, reward, done = Car0.step(random.uniform(0,30), random.uniform(-0.5*np.pi,0.5*np.pi))
        episode_reward = 0
        targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする

        while not terminal:
            i += 1
            V = 10

            action = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
            if action == 1:
                YR = 0.1
            elif action == 0:
                YR = -0.1
            else:
                YR = 0

            next_state, reward, done = Car0.step(V, YR)
            # print(next_state)
            drawer.plot_rectangle(Car0)
            # print('\r Episode:%4d, LoopTime:%4d' % (episode, i), end='')

            # 報酬を設定し、与える
            if done:
                terminal = True
                drawer.close_figure()

                next_state = np.zeros(state.shape)  # 次の状態s_{t+1}はない
                reward = 1  # maxstep超えて終了時は報酬
            else:
                reward = 0  # 各ステップで立ってたら報酬追加（はじめからrewardに1が入っているが、明示的に表す）

            episode_reward += 1 # reward  # 合計報酬を更新

            memory.add((state, action, reward, next_state))     # メモリの更新する
            state = next_state  # 状態更新

             # Qネットワークの重みを学習・更新する replay
            if (memory.len() > BATCH_SIZE) and not ISLEARNED:
                 mainQN.replay(memory, BATCH_SIZE, GAMMA, targetQN)

            if DQN_MODE:
                 targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする

            if i == MAXSTEP:
                terminal = True

                drawer.close_figure()

        if i == NUM_EPISODES:
            drawer.close_figure()
            print('')

    # env.sim(Car0, drawer)

if __name__ == '__main__':
    main()
