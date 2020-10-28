import os
import string
import unicodedata
import torch
import torch.nn.modules as nn

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
category_lines = {}
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
print('语言种类数为：{}'.format(len(category_lines.keys())))

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


# print(letterToTensor('D'))


# print(wordToTensor('DYH').size())

# 模型搭建
class RNN(nn.Module):
    #初始化 定义每一层的输入输出大小
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size,hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size,18)
        # self.softmax =nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size).cuda()

def invertToName(output):
    nameIndex = torch.topk(output,1).indices.item()
    name = all_categories[nameIndex]
    return name

# 训练数据抽样
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
rnn = RNN(n_letters,hidden_size,18).cuda()
# criterion = nn.NLLLoss()  # NLLLoss的结果就是把Label对应的那个输出值拿出来，再去掉负号，再求均值。
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss就是把Softmax–Log–NLLLoss合并成一步
lr = 0.005
# 测试
# input = letterToTensor('A')
# hidden = rnn.initHidden()
# output,hidden = rnn(input,hidden)
# category, word, category_tensor, word_tensor = randomTrainingSample()
# Loss = loss(output,category_tensor)
# print(output.size())
# print(category_tensor.size())
# print(Loss)
def train(word_tensor,catagory_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    output = torch.tensor([1,18]).cuda()

    # RNN的循环
    for i in range(word_tensor.size()[0]):
        output,hidden = rnn(word_tensor[i],hidden)

    #误差反向传播
    # print(output.size())
    # print(catagory_tensor.size())
    # print(output)
    # print(catagory_tensor.item())
    loss = criterion(output,catagory_tensor)
    loss.backward()

    #更新参数
    for p in rnn.parameters():
        p.data.add_(p.grad.data,alpha = -lr)

    return  output,loss.item()

epoch = 50000
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
    output = torch.tensor([1,18])
    for i in range(word_tensor.size()[0]):
        output,hidden = rnn(word_tensor[i],hidden)
    return output

with torch.no_grad():
    for i in range(1,epoch_test+1):
        category, word, category_tensor, word_tensor = randomTrainingSample()
        output = evaluate(word_tensor)
        output = invertToName(output)
        if output==category:
            correct= correct+1
        current_acc = correct/i
        if i%100 ==0:
            print('current acc is:{:.2f} '.format(current_acc))
    acc = correct / epoch_test
    print('acc in test is: ', acc)


#
# with open("all_losses", 'w') as f:
#     for i in all_loss:
#         f.write('{}\n'.format(i))
