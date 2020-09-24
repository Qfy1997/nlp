import numpy as np

def Forward(transition_probability,emission_probility,pi,obs_seq):
    """

    :param transition_probability: 状态转移矩阵——A（统计学习方法李航）
    :param emission_probility:概率发射矩阵——B
    :param pi:初识状态概率
    :param obs_seq:观察状态序列
    :return:F
    """
    transition_probability=np.array(transition_probability)
    emission_probility=np.array(emission_probility)
    print(emission_probility[:,0]) #前向算法输出
    pi=np.array(pi)
    Row=np.array(transition_probability).shape[0]

    F=np.zeros((Row,Col)) #最终的返回，公式中的alpha
    F[:,0]=pi*np.transpose(emission_probility[:,obs_seq[0]])
    print(F[:,0])
    for t in range(1,len(obs_seq)):
        for n in range(Row):
            F[n,t]=np.dot(F[:,t-1],transition_probability[:,n])*emission_probility[n,obs_seq[t]]

    return F

def Backward(transition_probability,emission_probability,pi,obs_seq):
    """

    :param transition_probability: 同上
    :param emission_probability:
    :param pi:
    :param obs_seq:
    :return:
    """
    transition_probability=np.array(transition_probability)
    emission_probability=np.array(emission_probability)
    pi=np.array(pi)

    Row=transition_probability.shape[0]
    Col=len(obs_seq)
    F=np.zeros((Row,Col))
    F[:,(Col-1):]=1

    for t in reversed(range(Col-1)):
        for n in range(Row):
            F[n,t]=np.sum(F[:,t+1]*transition_probability[n,:]*emission_probability[:,obs_seq[t+1]])

    return F

if __name__=='__main__':
    transition_probability=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]] #A
    emission_probability=[[0.5,0.5],[0.4,0.6],[0.7,0.3]] #B
    pi=[0.2,0.4,0.4]
    pi2=[0,0,0]
    #先用前向算法，在A、B、pi一致的前提下，求出特定观察序列的概率
    obs_seq=[0,1,0]
    Row=np.array(transition_probability).shape[0]
    Col=len(obs_seq)

    F=Forward(transition_probability,emission_probability,pi,obs_seq)
    F_back=Backward(transition_probability,emission_probability,pi2,obs_seq)
    res_forward=0
    for i in range(Row):
        res_forward+=F[i][Col-1]
    emission_probability=np.array(emission_probability)

    res_backward=0
    res_backward=np.sum(pi*F_back[:,0]*emission_probability[:,obs_seq[0]])
    print("前向算法：{}".format(res_backward))
    print("后向算法：{}".format(res_forward))