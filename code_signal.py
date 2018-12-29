import numpy as np
import matplotlib.pyplot as plt

# Behaviour of kron
# >>> np.kron([1,10,100], [5,6,7])
# array([  5,   6,   7,  50,  60,  70, 500, 600, 700])
# >>> np.kron([5,6,7], [1,10,100])
# array([  5,  50, 500,   6,  60, 600,   7,  70, 700])

def codeSignal(inVal, chipLen):
    # make ten copies of the signal
    bitWidth = 10

    # create a signal in range [-1,1] of length chipVals.
    str = format(inVal, 'b')
    dblVal = np.zeros(len(str))
    for i in range(len(str)):
        dblVal[i] = float(str[i])
    dblVal -= 0.5 # center on [-0.5,0.5]
    dblVal *= 2.0 # center on [-1,1]
    
    # zero pad the signal to be of length chips.
    if(len(dblVal) > chipLen):
        raise ValueError("length of signal > chipLen")
    padLen = chipLen - len(dblVal)
    dblVal = np.pad(dblVal, (padLen,0), 'constant', constant_values=(-1,0))

    # stretch out the signal.
    out = np.kron(dblVal, np.ones(bitWidth))
    return out