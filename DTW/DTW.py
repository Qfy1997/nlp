import numpy as np
import sys

#定义距离
def euc_dist(v1,v2):
    return np.abs(v1-v2)

# DTW的核心过程,实现动态规划
def dtw(s,t):
    """
    :param s: source sequence
    :param t: target sequence
    :return:
    """
    m,n=len(s),len(t)
    dtw = np.zeros((m,n))
    dtw.fill(sys.maxsize)

    #初始化过程
    dtw[0,0] = euc_dist(s[0],t[0])
    # print(dtw[0,0])
    for i in range(1,m):
        dtw[i,0]=dtw[i-1,0]+euc_dist(s[i],t[0])
    for i in range(1,n):
        dtw[0,i] = dtw[0,i-1]+euc_dist(s[0],t[i])

    #核心动态规划流程，此动态规划的过程依赖于上面的图
    for i in range(1,m):
        for j in range(max(1,i-10),min(n,i+10)): # local constraint
            cost=euc_dist(s[i],t[j])
            ds=[]
            ds.append(cost+dtw[i-1,j])
            ds.append(cost+dtw[i,j-1])
            ds.append(2*cost+dtw[i-1,j-1])
            ds.append(3*cost+dtw[i-1,j-2] if j>1 else sys.maxsize)
            ds.append(3*cost+dtw[i-2,j-1] if i>2 else sys.maxsize)

            dtw[i,j]=min(ds)

            # print(min(ds))

    return dtw[m-1,n-1]

print(dtw([3,6,9],[4,7,11]))