import numpy as np
import matplotlib.pyplot as plt

def testFuncPg4():
    n = np.arange(5,100)
    s = n**2
    print(n)
    print(s)
    plt.stem(s)

def testFuncPg5():
    ts = 1/2 #0
    t = np.arange(5, 15, ts)
    s = t**2
    plt.stem(t,s)
    plt.figure()
    plt.plot(t,s)
    plt.xlabel("my xlabel")
    plt.ylabel("my ylabel")

# pg 2/16
# pg 4 signal support and duration. Support range of non-zero values, duration is length of support.
# e.g. support from [0,3]

# 1.2.6 Signal Statistics
def testFuncPg6():
    testArray = [0, 1, 2, 3, 5]
    print(len(testArray))
    val = max(testArray)
    val = np.mean(testArray)
    # signal value distribution
    plt.hist(testArray, 2)
    val = np.histogram(testArray, 2)
    print(val)