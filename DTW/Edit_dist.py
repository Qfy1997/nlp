### Edit Distance (编辑距离)
# 编辑距离用来计算两个字符串之间的最短距离,这里涉及到三个不通过的操作, add,delete和replace.每一个操作我们假定需要1个单位的cost
# 例子："apple","appl"之间的编辑距离为1(需要1个删除的操作)
# spell correction
# "machine","macaide", dist=2
# "mach","aaach" dist=2
"""
s1 s2 s3 s4
t1 t2 t3 t4 t5
if s4==t5:
    return edit
"""
#基于动态规划的解法
def edit_dist(str1,str2):
    # m,n分别为字符串str1,str2的长度
    m,n = len(str1),len(str2)

    # 构建二位数组来储存子问题(sub-problem)的答案
    dp = [[0 for x in range(n+1)] for x in range(m+1)]

    # 利用动态规划算法,填充数组
    for i in range(m+1):
        for j in range(n+1):

            # 假设第一个字符串为空,则转换的代价为j(j次的插入)
            if i==0:
                dp[i][j]=j
            # 同样的,假设第二个字符串为空,则转换的代价为i(i次的插入)
            elif j == 0:
                dp[i][j]=i
            elif str1[i-1] == str2[j-1]:
                dp[i][j]=dp[i-1][j-1]

            # 如果最后一个字符不一样,则考虑多种可能性,并且选择其中最小的值
            else:
                dp[i][j] = 1 + min(dp[i][j-1],          #insert
                                   dp[i-1][j],          #remove
                                   dp[i-1][j-1])        #replace
    return dp[m][n]

print(edit_dist("apple","appazrt"))