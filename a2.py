import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from code_signal import codeSignal

class A2:
    @staticmethod
    def runAll():
        a2 = A2()
        # a2.q1()
        # a2.q2()
        # a2.q3()
        # a2.q4b()
        a2.q4c()
        pass

    def __init__(self):
        self.m_code1 = codeSignal(75,10)
        self.m_code2 = codeSignal(50,10)
        self.m_code3 = codeSignal(204,10)
        self.m_contents = sio.loadmat("A2/lab2_data.mat")
        pass
    
    def q1(self):
        plt.axis([1, 100, -1.5, 1.5])
        plt.subplot(1,3,1)
        plt.plot(self.m_code1)
        plt.subplot(1,3,2)
        plt.plot(self.m_code2)
        plt.subplot(1,3,3)
        plt.plot(self.m_code3)
        self.calMeanValAndEnergy(self.m_code1)
        self.calMeanValAndEnergy(self.m_code2)
        self.calMeanValAndEnergy(self.m_code3)
        self.calcCorrelation(self.m_code1, self.m_code2)
        self.calcCorrelation(self.m_code1, self.m_code3)
        self.calcCorrelation(self.m_code2, self.m_code3)
        self.calcCorrelation(self.m_code2, self.m_code2)
        plt.show()
    
    def q2(self):
        corr12 = self.runningCorrelation(self.m_code1, self.m_code2)
        plt.plot(corr12)
        corr33 = self.runningCorrelation(self.m_code3, self.m_code3)
        plt.plot(corr33)
        plt.show()

    
    def q3(self):
        # plot two signals.
        vals = self.m_contents['dsss']
        vals = np.squeeze(vals)
        vals = np.squeeze(vals)
        # vals = np.reshape(vals, len(vals))
        # print(vals)
        cs1 = codeSignal(170,10)
        self.calMeanValAndEnergy(cs1)
        cs2 = codeSignal(25,6)

        # compute the running correlation with cs1.
        corrDsssCs1 = self.runningCorrelation(vals, cs1)
        # corrDsssCs1Inv = self.runningCorrelation(cs1, vals)
        plt.plot(corrDsssCs1)
        plt.figure()
        # plt.plot(corrDsssCs1Inv)
        # plt.figure()

        subCorrCs1 = corrDsssCs1[len(cs1):len(vals) + len(cs1):len(cs1)]
        plt.plot(subCorrCs1)

        # compute the running correlation with cs1
        corrDsssCs2 = self.runningCorrelation(vals, cs2)
        subCorrCs2 = corrDsssCs2[len(cs2):len(vals) + len(cs2):len(cs2)]
        plt.plot(subCorrCs2)
        
        # plt.plot(vals)
        # plt.plot(cs1)
        # plt.plot(cs2)
        plt.show()
        pass

    def calMeanValAndEnergy(self, data):
        meanVal = np.mean(data)
        energyVal = np.sum(data**2)
        print("meanVal is=", meanVal, " meanEnergy is=", energyVal)
        pass
    
    def calcCorrelation(self, lhs, rhs):
        corr = np.sum(lhs * rhs)
        normFactor = np.sqrt(np.sum(lhs**2) + np.sum(rhs**2))
        normCorr = 1.0/normFactor * corr
        print("correlation=", corr, " norm correlation=", normCorr)
        pass

    def runningCorrelation(self, lhs, rhs):
        # we use the numpy correlation function.
        # out = np.zeros(len(lhs) + len(rhs) - 1)
        correlation = np.correlate(lhs, rhs, 'full')
        return correlation
    
    def q4b(self):
        radarPulse = self.m_contents["radar_pulse"]
        plt.subplot(1,2,1)
        plt.plot(radarPulse)
        # radarPulse = np.reshape(radarPulse, (radarPulse.size))
        radarPulse = np.reshape(radarPulse, -1)
        radarEnergy = np.sum(radarPulse**2)
        print("radar energy is ", radarEnergy)
        radarReceived = self.m_contents["radar_received"]
        radarReceived = np.squeeze(radarReceived)
        plt.subplot(1,2,2)
        plt.plot(radarReceived)
        plt.figure()

        radarCorr = self.runningCorrelation(radarReceived, radarPulse)
        plt.plot(radarCorr)
        plt.show()
        pass

    def q4c(self):
        radarNoise = self.m_contents["radar_noise"]
        radarNoise = np.squeeze(radarNoise)
        pulseEnergy = 37.5
        # plt.plot(radarNoise)

        radarPulse = self.m_contents["radar_pulse"]
        radarPulse = np.squeeze(radarPulse)
        radarCorr = self.runningCorrelation(radarNoise, radarPulse)
        threshold = np.where(radarCorr > (pulseEnergy)/2, 1, -1)
        # plt.plot(threshold)
        # plt.figure()
        # plt.plot(radarCorr)
        # plt.show()

        # estimate the false alarm rate and the miss rate.
        # 4c (pg 13, last page)

        # false alarm rate
        print("number of samples is ", threshold.size)
        aboveThreshold = np.where(threshold > 0)[0]
        falseAlarmRate = len(aboveThreshold)/threshold.size
        print("false alarm rate is ", falseAlarmRate)

        # miss rate.
        radarCorrPlusSignal = radarCorr + pulseEnergy
        thresholdMiss = np.where(radarCorrPlusSignal < (pulseEnergy)/2, 1, -1)
        aboveThreshold = np.where(thresholdMiss > 0)[0]
        missRate = len(aboveThreshold)/thresholdMiss.size
        print("miss rate is ", missRate)

        # plt.plot(radarCorrPlusSignal)
        # plt.plot(thresholdMiss)
        # plt.show()

        # plot histogram.
        plt.plot(radarCorr)
        plt.figure()
        plt.hist(radarCorr, 100)
        plt.show()

        # recompute the false alarm rate and miss rate using histogram.
        # false alarm rate
        hist, binEdges = np.histogram(radarCorr,5000)
        numElements = np.sum(hist)
        print("num histogram elements is ", numElements)
        firstIndexGreater = np.argmax(binEdges > pulseEnergy / 2)
        print("first index greater is ", firstIndexGreater)
        falseAlarmElements = np.sum(hist[firstIndexGreater:])
        print("false alarm elements is ", falseAlarmElements)
        print("false alarm is ", falseAlarmElements/numElements)

        # miss rate
        firstIndexLess = np.argmin(binEdges < (pulseEnergy / 2 - pulseEnergy))
        data = binEdges < (pulseEnergy / 2 - pulseEnergy)
        print("data is ", data, " with length ", len(data))
        print("first index less is ", firstIndexLess)
        missElements = np.sum(hist[0:firstIndexLess])
        print("missRate is ", missElements/numElements)

        # total error rate = miss rate + false alarm rate.
        pass

        def test(self):