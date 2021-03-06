import numpy as np

def IsInCollision(x,obc):
    sizes = [(5,5,10),(5,10,5),(5,10,10), (10,5,5), (10,5,10), (10,10,5), (10,10,10), (5,5,5), (10,10,10), (5,5,5)]
    s=np.zeros(3,dtype=np.float32)
    s[0]=x[0]
    s[1]=x[1]
    s[2] = x[2]
    for i in range(0,10):
        cf=True
        for j in range(0,3):
            if abs(obc[i][j] - s[j]) > sizes[i][j]/2.0 and s[j]<20.0 and s[j]>-20:
                cf=False
                break
        if cf==True:
            return True
    return False
