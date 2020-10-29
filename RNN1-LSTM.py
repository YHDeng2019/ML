import os
import string
import unicodedata
import torch
import torch.nn.modules as nn

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
category_lines = {}  # 18类，共20074条
all_categories =[]
path = 'data/names'

# invert unicode to ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

# 把每一门语言换成ascii码
def readline(filepath):
    fr = open(filepath)
    lines = fr.readlines()
    return [unicodeToAscii(line.strip()) for line in lines]


def readfile(path):
    languages = os.listdir(path)
    for (i, language) in enumerate(languages):
        all_categories.append(language.split('.')[0])
        filepath = os.path.join(path, language)
        category_lines[language.split('.')[0]] = readline(filepath)


readfile(path) # 18种语言
num_values = 0
for i in category_lines.values():
    num_values+=len(i)
print('语言种类数为：{} ,共有{}条数据'.format(len(category_lines.keys()),num_values))

# print (category_lines.keys())
# print (len(category_lines))
# print (category_lines['Chinese'])

# 查找每个字母在字母表中的index
def findLetterIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    letterTensor = torch.zeros(1, n_letters)
    letterTensor[0][findLetterIndex(letter)] = 1
    return letterTensor.cuda()


def wordToTensor(word):
    lenth = len(word)
    wordTensor = torch.zeros(lenth, 1, n_letters)
    for i, letter in enumerate(word):
        wordTensor[i] = letterToTensor(letter)
    return wordTensor.cuda()


"""
模型定义（LSTM）
"""
class LSTM(nn.Module):
    #初始化 定义每一层的输入输出大小
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size)  # (57,18)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size,18)
        )

    def forward(self,input,hidden,cell):
        output,(hidden,cell) = self.lstm(input,(hidden,cell))  # input(seq_len,batch,input_size) output(seq_len,batch,hidden_size)
        output = self.fc(output)  # 增加全连接层会使模型效果更好
        return output,(hidden,cell)

    def initHidden(self):
        # h0 = torch.randn(self.num_layers,2, 1, 18)
        return  torch.zeros(1,1,self.hidden_size).cuda() # (num_layers,batch,hidden_size)

    def initCell(self):
        return torch.zeros(1, 1,self.hidden_size).cuda() # (num_layers,batch,hidden_size)

def invertToName(output):
    nameIndex = torch.topk(output,1).indices.item()
    name = all_categories[nameIndex]
    return name

'''
随机抽样训练数据
'''
import random
def randomChoice(l):
    return l[random.randint(0,len(l)-1)]

# 训练数据抽样
def randomTrainingSample():
    category = randomChoice(all_categories)
    word = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long).cuda()
    word_tensor = wordToTensor(word)
    return category, word, category_tensor, word_tensor


'''
训练模型
'''
hidden_size =128
rnn = LSTM(n_letters,hidden_size).cuda()
# criterion = nn.NLLLoss()  # NLLLoss的结果就是把Label对应的那个输出值拿出来，再去掉负号，再求均值。
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss就是把Softmax–Log–NLLLoss合并成一步
lr = 0.001
optim = torch.optim.Adam(rnn.parameters(),lr=lr)

def train(word_tensor,catagory_tensor):
    rnn.train()

    hidden = rnn.initHidden()
    cell = rnn.initCell()

    rnn.zero_grad()
    optim.zero_grad()

    # LSTM的循环
    output, (hidden, cell) = rnn(word_tensor, hidden, cell) # [n,1,18] output存放的是每一层的输出 ,detach防止梯度爆炸
    last_output = output[-1]                                # [1,18]  最后的output等于最后一层的hidden

    #误差反向传播
    loss = criterion(last_output,catagory_tensor)
    loss.backward()

    #更新参数
    # for p in rnn.parameters():
    #     p.data.add_(p.grad.data,alpa = -lr)

    # 2
    optim.step()

    return  last_output,loss.item()

epoch = 20000
current_loss = 0
all_loss =  []
for i in range(1,epoch+1):
    category, word, category_tensor, word_tensor = randomTrainingSample()
    output,loss = train(word_tensor,category_tensor)
    output = invertToName(output)
    current_loss += loss
    if i%1000 == 0:
        correct = 'YES' if category==output else 'NO'
        print('{0:^7} {1:^3}%  Loss:{2:.2f} {3:^12}  {4:<10}-{5:<10}  {6:<3}'.format(i,i*100//epoch,loss,word,category,output,correct.rjust(5)))
    if i % 100 == 0:
        all_loss.append(current_loss/i)


'''
测试模型
'''
epoch_test = 10000
correct = 0
acc = 0.0
def evaluate(word_tensor):
    hidden = rnn.initHidden()
    cell = rnn.initCell()
    output, (hidden, cell) = rnn(word_tensor, hidden, cell)
    return output[-1]

with torch.no_grad():
    for i in range(1,epoch_test+1):
        category, word, category_tensor, word_tensor = randomTrainingSample()
        output = evaluate(word_tensor)
        output = invertToName(output)
        if output==category:
            correct= correct+1
        current_acc = correct/i
        if i%100 ==0:
            print('current acc is:{:.2f} 正确个数：{} '.format(current_acc,correct))
    acc = correct / epoch_test
    print('acc in test is: ', acc)


#
# with open("all_losses", 'w') as f:
#     for i in all_loss:
#         f.write('{}\n'.format(i))
