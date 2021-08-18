import pandas as pd
import numpy as np

t = pd.read_csv("local/sentiment_train_test_header.csv")


TEXT = t['5']
LABEL = t['0']

neg_l=[]
neg_t=[]
pos_l=[]
pos_t=[]


for idx, l in enumerate(LABEL):
    if l == 0:
        neg_l.append(LABEL[idx])
        neg_t.append(TEXT[idx])
    elif l==4:
        pos_l.append(LABEL[idx])
        pos_t.append(TEXT[idx])

print(len(neg_t),len(neg_l),len(pos_t),len(pos_l))

LABEL1=[]
TEXT1=[]
LABEL2=[]
TEXT2=[]

for i in range(0,50000):
    LABEL1.append(neg_l[i])
    LABEL1.append(pos_l[i])
    TEXT1.append(neg_t[i])
    TEXT1.append(pos_t[i])

for i in range(50000,100000):

    LABEL2.append(neg_l[i])
    LABEL2.append(pos_l[i])
    TEXT2.append(neg_t[i])
    TEXT2.append(pos_t[i])


data1 = pd.DataFrame({
    'Label':LABEL1,
    'Text':TEXT1
})
print(data1[:1])
data1.to_csv("./local/sentiment_train_test_header_1.csv", index=False)

data2 = pd.DataFrame({
    'Label':LABEL2,
    'Text':TEXT2
})
print(data2[:1])
data1.to_csv("./local/sentiment_train_test_header_2.csv", index=False)