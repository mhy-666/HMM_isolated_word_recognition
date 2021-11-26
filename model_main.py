from python_speech_features import *
from scipy.io import wavfile
import joblib
import numpy as np
import os
import re
import HMMModel
import wave


# 生成wav字典并作为返回值返回
def generate_wav(wavpath):
    dictionaryWav = {}
    dictionaryLabel = {}
    for (dirpath, dirnames, filenames) in os.walk(wavpath):
        for filename in filenames:
            if filename.endswith('.wav'):
                WaveFile = os.sep.join([dirpath, filename])
                fileid = WaveFile
                dictionaryWav[fileid] = WaveFile
                label=re.split(r'[/\\_]\s*', WaveFile)[7]
                label=label[0]
                dictionaryLabel[fileid] = label
    return dictionaryWav, dictionaryLabel
# 提取MFCC特征值
def getMFCC(file):
    fs, audio = wavfile.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=13, winlen=0.025, winstep=0.01, nfilt=26, nfft=2048, lowfreq=0,
                     highfreq=None, preemph=0.97)
    d_mfcc_feat = delta(mfcc_feat, 1)
    d_mfcc_feat2 = delta(mfcc_feat, 2)
    feature_mfcc = np.hstack((mfcc_feat, d_mfcc_feat,d_mfcc_feat2))
    #最终得到一个39维的特征向量
    return feature_mfcc

class TrainModel():
    def __init__(self, words=None, state_num=15, cov_type='full', itertimes_num=20):
        super(TrainModel, self).__init__()
        print(words)
        self.words = words
        self.category = len(words)
        self.n_mix = state_num
        self.cov_type = cov_type
        self.n_iter = itertimes_num
        # models为11个模型的容器
        self.models = []
        for k in range(self.category):
            model = HMMModel.HMM_model.GMMHMM(1,15)
            self.models.append(model)

    # 模型训练
    def train(self, wavdict=None, labeldict=None):
        count=np.zeros(11)
        for k in range(11):
            for x in wavdict:
                if labeldict[x] == self.words[k]:
                    mfcc_featSingle = getMFCC(wavdict[x])
                    count[k]=count[k]+mfcc_featSingle.shape[0]
            print(count)
        print("样本已分离")
        samples_1 = np.zeros((int(count[0]), 39))
        samples_2=  np.zeros((int(count[1]), 39))
        samples_3 = np.zeros((int(count[2]), 39))
        samples_4 = np.zeros((int(count[3]), 39))
        samples_5 = np.zeros((int(count[4]), 39))
        samples_6 = np.zeros((int(count[5]), 39))
        samples_7 = np.zeros((int(count[6]), 39))
        samples_8 = np.zeros((int(count[7]), 39))
        samples_9 = np.zeros((int(count[8]), 39))
        samples_O = np.zeros((int(count[9]), 39))
        samples_Z = np.zeros((int(count[10]), 39))
        for k in range(11):
            for x in wavdict:
                if labeldict[x] == self.words[k]:
                    if (k == 0):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_1 = np.vstack((samples_1, mfcc_featSingle))
                    if (k == 1):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_2 = np.vstack((samples_2, mfcc_featSingle))
                    if (k == 2):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_3 = np.vstack((samples_3, mfcc_featSingle))
                    if (k == 3):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_4 = np.vstack((samples_4, mfcc_featSingle))
                    if (k == 4):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_5 = np.vstack((samples_5, mfcc_featSingle))
                    if (k == 5):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_6 = np.vstack((samples_6, mfcc_featSingle))
                    if (k == 6):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_7 = np.vstack((samples_7, mfcc_featSingle))
                    if (k == 7):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_8 = np.vstack((samples_8, mfcc_featSingle))
                    if (k == 8):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_9 = np.vstack((samples_9, mfcc_featSingle))
                    if (k == 9):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_O = np.vstack((samples_O, mfcc_featSingle))
                    if (k == 10):
                        mfcc_featSingle = getMFCC(wavdict[x])
                        samples_Z = np.vstack((samples_Z, mfcc_featSingle))
            if (k == 0):
                model = self.models[k]
                model.fit(samples_1)
            if (k == 1):
                model = self.models[k]
                model.fit(samples_2)
            if (k == 2):
                model = self.models[k]
                model.fit(samples_3)
            if (k == 3):
                model = self.models[k]
                model.fit(samples_4)
            if (k == 4):
                model = self.models[k]
                model.fit(samples_5)
            if (k == 5):
                model = self.models[k]
                model.fit(samples_6)
            if (k == 6):
                model = self.models[k]
                model.fit(samples_7)
            if (k == 7):
                model = self.models[k]
                model.fit(samples_8)
            if (k == 8):
                model = self.models[k]
                model.fit(samples_9)
            if (k == 9):
                model = self.models[k]
                model.fit(samples_O)
            if (k == 10):
                model = self.models[k]
                model.fit(samples_Z)


    # 对测试wav字典使用维特比算法，然后得到最大概率模型所对应的标签
    def viterbi(self, filepath):
        result = []
        for k in range(self.category):
            model = self.models[k]
            data1 = []
            data2 = []
            mfcc_feat = getMFCC(filepath)
            destination = model.score(mfcc_feat)
            data1.append(destination)
            result.append(data1)
            #print(destination)

        # 通过维特比算法得到最大概率模型所对应的标签
        result = np.vstack(result).argmax(axis=0)
        result = [self.words[label] for label in result]
        #print('得到的标签为：\n',result)
        return result

    # 用external
    # joblib保存生成的hmm模型
    def saveModels(self, path="M:/pycharmProjects/new/HMM/models/440-15-20iter_models.pkl"):
        joblib.dump(self.models, path)

    # 用external
    # joblib导入hmm模型
    def loadModels(self, path="M:/pycharmProjects/new/HMM/models/440-15-20iter_models.pkl"):
        self.models = joblib.load(path)

    def test(self,wavOfTest,labelOfTest):
        trueCount = 0
        falseCount = 0
        for k in wavOfTest:
            wav_path = wavOfTest[k]
            res = models.viterbi(wav_path)[0]
            #print(wavOfTest[k], res, labelOfTest[k])
            if res == labelOfTest[k]:
                trueCount += 1
            else:
                falseCount += 1
        print(trueCount, falseCount)
        print("识别准确率为:", trueCount / (trueCount + falseCount))

#开始测试

if __name__ == '__main__':
    dictionaryWords = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'O', 'Z']
    wavOfTrain, labelOfTrain = generate_wav("M:/pycharmProjects/new/HMM/wavoftrain")
    wavOfTest, labelOfTest = generate_wav("M:/pycharmProjects/new/HMM/wavoftest")
    # # 开始训练模型
    models = TrainModel(words=dictionaryWords)
    print("start trainging....")
    models.train(wavOfTrain,labelOfTrain)
    print("finish trainging....")
    models.saveModels()
    models = TrainModel(words=dictionaryWords)
    models.loadModels()
    print('test begin!')
    models.test(wavOfTest,labelOfTest)