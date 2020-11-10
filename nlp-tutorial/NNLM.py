import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split() # space tokenizer
        input = [word_dict[n] for n in word[:-1]] # create (1~n-1) as input
        target = word_dict[word[-1]] # create (n) as target, We usually call this 'casual language model'

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# Model
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m) #词典大小n_class,m为嵌入向量的维度，即用多少维来表示一个向量
        self.H = nn.Linear(n_step * m, n_hidden, bias=False) #用于设置网络中的全连接层，需要注意的是全连接层的输入与输出都是二维张量,输入为n_step*m,输出为n_hidden
        self.d = nn.Parameter(torch.ones(n_hidden)) #parameter为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到moudle里面，在参数优化时可以进行优化。
        """
        .parameter():类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        """
        self.U = nn.Linear(n_hidden, n_class, bias=False)
        self.W = nn.Linear(n_step * m, n_class, bias=False)
        self.b = nn.Parameter(torch.ones(n_class))

    def forward(self, X):
        X = self.C(X) # X : [batch_size, n_step, n_class]
        #print(X)
        X = X.view(-1, n_step * m) # [batch_size, n_step * n_class] view()改变维度，参数-1可以理解为这一层的维度不用算
        tanh = torch.tanh(self.d + self.H(X)) # [batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh) # [batch_size, n_class]
        return output

if __name__ == '__main__':
    n_step = 2 # number of steps, n-1 in paper 词代码中sentence中每一句话三个单词，这里指一句话中的前两个词
    n_hidden = 2 # number of hidden size, h in paper, hidden中的元神经元个数
    m = 2 # embedding size, m in paper，用m维度的向量来表示一个向量

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # number of Vocabulary

    model = NNLM()
    print(model.C)
    print(model.H)
    print("D:")
    print(model.d)
    print(model.b)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)
    target_batch = torch.LongTensor(target_batch)
    print(input_batch)
    # Training
    for epoch in range(5000):
        optimizer.zero_grad()
        """
        显然，我们进行下一次batch梯度计算的时候，
        前一个batch的梯度计算结果，没有保留的必要了。
        所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0。
        """
        output = model(input_batch)
        #if(epoch==0):
            #print(output)
        # output : [batch_size, n_class], target_batch : [batch_size]
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward() #反向传播
        optimizer.step() #更新参数

    # Predict
    predict = model(input_batch).data.max(1, keepdim=True)[1]
    print(word_dict)
    print(model(input_batch).data)
    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])