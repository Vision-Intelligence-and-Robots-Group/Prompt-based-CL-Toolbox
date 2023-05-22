
import random
import os
root_path = "/home/wangyabin/workspace/datasets/domainnet"

domainname = ['clipart','infograph','painting','quickdraw','real','sketch']
# random session
choseDomain = ['clipart','infograph','painting','quickdraw','real','sketch','clipart','infograph','painting','quickdraw','real','sketch']
random.shuffle(choseDomain)
choseDomain = choseDomain[:10]
print(choseDomain)
import pdb;pdb.set_trace()


chosenClass = list(range(200))
AllClass = list(range(345))

traindict = { i:{} for i in domainname }
for domain in domainname:
    for cls in AllClass:
        traindict[domain][cls]=[]

testdict = { i:{} for i in domainname }
for domain in domainname:
    for cls in AllClass:
        testdict[domain][cls]=[]


for domain in domainname:
    with open(os.path.join(root_path, domain+"_train.txt"), "r") as f:
        data = f.readlines()

    for line in data:
        line = line.split()
        path = line[0]
        cls = int(line[1])
        traindict[domain][cls].append(path)

for domain in domainname:
    with open(os.path.join(root_path, domain+"_test.txt"), "r") as f:
        data = f.readlines()

    for line in data:
        line = line.split()
        path = line[0]
        cls = int(line[1])
        testdict[domain][cls].append(path)






ImgsPerCls = { i:[] for i in AllClass }
for domain in domainname:
    for cls in AllClass:
        ImgsPerCls[cls].extend(traindict[domain][cls])

numImgsPerCls = [ len(ImgsPerCls[i]) for i in AllClass ]


import numpy as np

npnumImgsPerCls = np.array(numImgsPerCls)

k_largeCls = (-npnumImgsPerCls).argsort()[:200]


np.random.shuffle(k_largeCls)


train_set = []
test_set = []
total_sessions = 10
clsnum_session = 20

for session in range(total_sessions):
    domain = choseDomain[session]
    for cls in range(clsnum_session):
        index = session*clsnum_session+cls
        truecls = k_largeCls[index]
        aligncls = index
        for path in traindict[domain][truecls]:
            train_set.append([path, aligncls])


for session in range(total_sessions):
    domain = choseDomain[session]
    for cls in range(clsnum_session):
        index = session*clsnum_session+cls
        truecls = k_largeCls[index]
        aligncls = index
        for path in testdict[domain][truecls]:
            test_set.append([path, aligncls])



f=open('train.txt', "a+")
for i in train_set:
    new_context = i[0]+ ' ' + str(i[1]) + '\n'
    f.write(new_context)

f=open('test.txt', "a+")
for i in test_set:
    new_context = i[0]+ ' ' + str(i[1]) + '\n'
    f.write(new_context)




