import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sig
from signal_detector import SignalDetector

class A1:
    @staticmethod
    def runAll():
        # a1 = A1()
        # a1.runTest()
        # a1.q1()
        # a1.q1c()
        # a1.q2()
        # a1.q3()
        signalDetector = SignalDetector(512, 0.01)
        contents = sio.loadmat("A1/lab1_data.mat")
        signal = contents["mboc"]
        plt.plot(np.linspace(0,1,len(signal)),signal)

        signalDetector.detect(signal)
        plt.show()

    def __init__(self):
        pass
    
    def run(self):
        print("Running A1")
    
    def q1(self):
        #1A
        print("Start Q1")
        n = np.arange(1,50)
        s = np.sin(2*np.pi*n/50)
        plt.plot(n, s)

        #1B
        # calculate the maximum value
        maxIndex = np.argmax(s)
        maxVal = s[maxIndex]
        print("max value occurs at ", maxIndex)
        print("max value has value ", maxVal)

        # calculate the minimum value
        minVal = np.min(s)
        meanVal = np.mean(s)
        print("min val is ", minVal)
        print("mean val is ", meanVal)
        meanSqVal = np.mean(s**2)
        rmsVal = np.sqrt(meanSqVal)
        print("mean squared is", meanSqVal)
        print("rms is", rmsVal)

        # energy is 
        energyVal = np.sum(s**2)
        print("energy is", energyVal)

        # calculate the min value
        print("Running Q1")
        plt.show()
    
    def q1c(self):
        # suppose that s is the result of sample a continuous time signal
        # with interval Ts = 1/100. Use discrete-time statistics to estimate 
        # continuous signal s(t) = sin(4*pi*t)

        # signal duration
        n = np.arange(0, 1, 1/100)
        s = np.sin(4*np.pi*n)
        plt.plot(n,s)
        # plt.show()

        # energy
        energyVal = np.sum(s**2)
        print("energy is", energyVal)

        # average power
        averagePower = np.mean(s**2)
        print("averager power is ", averagePower)

        # rms value.
        rmsVal = np.sqrt(averagePower)
        print("rms val is ", rmsVal)
        pass

    def q2(self):
        contents = sio.loadmat("lab1_data.mat")
        clarinet = contents['clarinet']

        # a-c
        if False:
            print(clarinet)
            n = np.arange(0, len(clarinet))
            plt.plot(n, clarinet)

        #d
        if False:
            plt.hist(clarinet, 50)
            plt.show()
        
        # print("Hello World")
        #e
        meanVal = np.mean(clarinet)
        energyVal = np.sum(clarinet**2)
        meanSqVal = np.mean(clarinet**2)
        rmsVal = np.sqrt(meanSqVal)
        print("mean val is ", meanVal)
        print("energy val is ", energyVal)
        print("meanSqVal is ", meanSqVal)
        print("rmsVal is ", rmsVal)
        pass

    def lab1Sys1(self, input):
        rand = np.random.rand(len(input), 1) * 0.2
        # plt.stem(rand)
        # plt.figure()
        # rand = np.random.uniform(low=-1.0, high=1.0, size=(len(input), 1))
        output = rand + input
        return output
    
    def lab1Sys2(self, input):
        numerator = np.ones((10,1)) / 10        # numerator is 10,1
        # print("length of input is", input.shape) # shape is 100,1

        out = np.squeeze(numerator)              # squeeze gives 10

        out2 = np.reshape(out, (len(out), 1))    # reshape gives 10,1

        out3 = np.reshape(out, (len(out)))       # reshape gives 10

        output = sig.filtfilt(out, np.ones(1), np.squeeze(input))
        # plt.figure()
        return output 

    def q3(self):
        contents = sio.loadmat("lab1_data.mat")
        clarinet = contents['clarinet']
        clarinet = clarinet[0:100]
        print("length of clarinet is ", len(clarinet))

        sys1out = self.lab1Sys1(clarinet)
        mse = np.mean((sys1out-clarinet)**2)
        print("mse is ", mse)                       # compute the mean squared error
        print("rmse is ", np.sqrt(mse))             # compute the root mean squared error

        sys2out = self.lab1Sys2(clarinet)
        mse = np.mean((sys2out-clarinet)**2)
        print("mse is ", mse)                       # compute the mean squared error
        print("rmse is ", np.sqrt(mse))             # compute the root mean squared error

        # compute the ms and rms values as
        # mse = mean((s-s_mod)^2)
        # rmse = sqrt(mse)
        plt.plot(sys1out) 
        plt.plot(sys2out) 
        plt.plot(clarinet)
        # plt.plot(sys1out) #len(sys1out), sys1out)
        plt.show()
