import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class SignalDetector:
    def __init__(self, blockSize, threshold):
        self.m_blockSize = blockSize
        self.m_threshold = threshold
        pass


    # input is matrix
    def detect(self, input):
        print("input matrix is", input.shape)
        numBlocks = int(np.ceil(len(input)/self.m_blockSize))
        paddedInput = input.copy()
        paddedInput.resize((numBlocks * self.m_blockSize, 1))
        print("numBlocks is ", numBlocks)
        print("paddedInput is ", paddedInput.shape)
        paddedInput = np.reshape(paddedInput, (-1, self.m_blockSize))
        # np.reshape(paddedInput, (numBlocks, self.m_blockSize))

        detectResults = []
        for i in range(0, len(paddedInput)):
            detectResult = self.detectBlock(paddedInput[i])
            detectResults.append(detectResult)
        
        plt.plot(np.linspace(0,1,len(detectResults)), detectResults)
        pass

    def detectBlock(self, block):
        # compute the rms value
        meanSqVal = np.mean(block**2)
        detected = meanSqVal > self.m_threshold
        return detected