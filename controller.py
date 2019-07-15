# taken from https://github.com/normandipalo/learn-to-walk-with-genetic-algs
import numpy as np
sin = np.sin

def output(a,T,t):
    # Output of a 4th degree Fourier Series of sin.
    # INPUT: the 4 harmonics weights, time period T, and the time t.
        y=0
        
        for i in range(4):
            y+=a[i]*sin((i+1)*np.pi*2*t/T+a[i+4])
        return y

def input(w,T,t):
    #This function generates the input to the model.
    #INPUT: weights, time step
    #Returns: input arrat
    inputs=[]
    
    #The model input has dimension 18. Eventhough, I only use 9 functions, since the last 9 are just the same but with a -
    #When one of the inputs is <0 it is =0 for the model.

    inputs=[-output(w[0],T,t),output(w[1],T,t),-output(w[2],T,t), 
    output(w[3],T,t),output(w[4],T,t),output(w[5],T,t),
    -output(w[6],T,t),-output(w[7],T,t),output(w[8],T,t),
    output(w[0],T,t+T/2),-output(w[1],T,t+T/2),output(w[2],T,t+T/2), 
    -output(w[3],T,t+T/2), -output(w[4],T,t+T/2), -output(w[5],T,t+T/2),
     output(w[6],T,t+T/2), output(w[7],T,t+T/2), -output(w[8],T,t+T/2),]

    return inputs