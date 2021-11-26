import numpy as np
from math import pi,sqrt,exp,pow,log
from numpy.linalg import det, inv
from hmmlearn import hmm as HMM_model
from abc import ABCMeta, abstractmethod
from sklearn import cluster



#定义连续分布即高斯分布下的HMM模型，这是因为我们的音频特征值属于连续的向量，而它又作为HMM模型的观测值。
class GMMHMM:
    def __init__(self, state_num=1, x_len=1, itertimes_num=20):
        self.state_num = state_num
        self.itertimes_num = itertimes_num
        self.x_len = x_len
        # 初始化观测概率均值
        self.means_B = np.zeros((state_num, x_len))
        # 初始化观测概率协方差
        self.covars_B = np.zeros((state_num, x_len, x_len))
        # 初始化为均值为0，方差为1
        for i in range(state_num): self.covars_B[i] = np.eye(x_len)

    def _init(self,X):
        # 通过K均值聚类，确定状态初始值
        Kofmeans = cluster.KMeans(n_clusters=self.state_num)
        Kofmeans.fit(X)
        self.means_B = Kofmeans.cluster_centers_
        for i in range(self.state_num):
            self.covars_B[i] = np.cov(X.T) + 0.01 * np.eye(len(X[0]))

    # 求x在状态k下的观测概率
    def prob_B(self, x):
        prob = np.zeros((self.state_num))
        for i in range(self.state_num):
            prob[i]=gauss2D(x, self.means_B[i], self.covars_B[i])
        return prob

    # 根据状态生成x p(x|z)
    def generate_xList(self, z):
        return np.random.multivariate_normal(self.means_B[z][0], self.covars_B[z][0], 1)

    # 更新观测概率
    def update_prob_B(self, X, post_state):
        for k in range(self.state_num):
            for j in range(self.x_len):
                self.means_B[k][j] = np.sum(post_state[:, k] * X[:, j]) / np.sum(post_state[:, k])

            X_cov = np.dot((X - self.means_B[k]).T, (post_state[:, k] * (X - self.means_B[k]).T).T)
            self.covars_B[k] = X_cov / np.sum(post_state[:, k])
            if det(self.covars_B[k]) == 0: # 对奇异矩阵的处理
                self.covars_B[k] = self.covars_B[k] + 0.01 * np.eye(len(X[0]))

    # 对观测序列的训练
    def fit(self, X, Z_seq=np.array([])):
        # 输入X观测序列
        # 输入Z为未知的状态序列，待求
        self.trained = True
        X_length = len(X)
        self._init(X)

        # 对状态序列进行初始化
        if Z_seq.any():
            Z = np.zeros((X_length, self.state_num))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.state_num))
        # EM步骤迭代
        for e in range(self.itertimes_num):
            print(e, " iter")
            # 求期望即最大似然
            # 向前向后传递因子
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.state_num, self.state_num))  # 相邻状态的联合后验概率
            for i in range(X_length):
                if i == 0: continue
                if c[i] == 0: continue
                post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1], beta[i] * self.prob_B(X[i])) * self.transmat_prob

            # 有了最大似然，可以获得估计参数
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.state_num):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.update_prob_B(X, post_state)

    # 利用HMM前向算法求前向概率
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.state_num))  # P(x,z)
        alpha[0] = self.prob_B(X[0]) * self.start_prob * Z[0]  # 初始值
        # 归一化因子
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # 递归传递
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.prob_B(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i] == 0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    #利用后向算法求后向概率
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.state_num))  # P(x|z)
        beta[X_length - 1] = np.ones((self.state_num))
        # 递归传递
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.prob_B(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i + 1] == 0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta

    #利用维特比算法，已知序列求其隐藏状态值，最后得到最大概率
    def viterbi(self, X, istrain=True):
        if self.trained == False or istrain == False:  # 需要根据该序列重新训练
            self.fit(X)

        X_length = len(X)  # 序列长度
        state = np.zeros(X_length)  # 隐藏状态

        pre_state = np.zeros((X_length, self.state_num))  # 保存转换到当前隐藏状态的最可能的前一状态
        max_pro_state = np.zeros((X_length, self.state_num))  # 保存传递到序列某位置当前状态的最大概率

        _,c=self.forward(X, np.ones((X_length, self.state_num)))
        max_pro_state[0] = self.prob_B(X[0]) * self.start_prob * (1 / c[0]) # 初始概率

        # 前向求和的概率
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.state_num):
                prob_state = self.prob_B(X[i])[k] * self.transmat_prob[:, k] * max_pro_state[i - 1]
                max_pro_state[i][k] = np.max(prob_state)* (1/c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # 后向求和的概率
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1,:])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return  state

    def expand_list(self, X):
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # 生成状态序列和观测序列
    def generate_seq(self, seq_length):
        X = np.zeros((seq_length, self.x_len))
        Z = np.zeros(seq_length)
        Z_pre = np.random.choice(self.state_num, 1, p=self.start_prob)
        # 获得采样的初值
        X[0] = self.generate_xList(Z_pre)
        Z[0] = Z_pre
        for i in range(seq_length):
            if i == 0: continue
            Z_next = np.random.choice(self.state_num, 1, p=self.transmat_prob[Z_pre, :][0])
            Z_pre = Z_next
            X[i] = self.generate_xList(Z_pre)
            Z[i] = Z_pre
        return X, Z


# 二元高斯分布函数，求观测概率B（ij）的时候需要用到。
def gauss2D(x, mean, cov):
    z = -np.dot(np.dot((x-mean).T,inv(cov)),(x-mean))/2.0
    temp = pow(sqrt(2.0*pi),len(x))*sqrt(det(cov))
    return (1.0/temp)*exp(z)
