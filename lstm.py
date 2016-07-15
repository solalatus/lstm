import numpy as np


def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))


def softmax(w, t = 1.0):
    print("----Inside Softmax----")
    e = np.exp(np.array(w) / t)
    print(e)
    dist = e / np.sum(e)
    print(dist)
    print("--- Exit Softmax ---")
    return dist

word_dim = 5
hidden_dim = 2
#bptt_truncate = bptt_truncate

vocab=np.arange(word_dim)

U_in = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_fo = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_ou = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))

#for softmax
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))

W_in = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_fo = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_ou = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_g = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


x=np.array([[0,0,0,0,1],[1,0,0,0,0],[0,0,1,0,0]])
y=np.array([[0,0,0,0,0],[1,0,0,0,0],[0,0,1,0,0]])
y_hat=[]
s=np.zeros(hidden_dim)
c=np.zeros(hidden_dim)

print("weight U: ", U_in)
print("weight W: ", W_in)
print("---------------init done-------------")
for iter1 in range(len(x)):
    print("input layer calculation")
    print("input: ", x[iter1], " ")

    print("-------")
    print(iter1)
    print("---------------------------------------")

    print("-- i --")
    i = sigmoid( np.dot(U_in, x[iter1]) + np.dot(W_in, s) )
    print(i)

    print("-- f --")
    f = sigmoid( np.dot(U_fo, x[iter1]) + np.dot(W_fo, s) )
    print(f)

    print("-- o --")
    o = sigmoid( np.dot(U_ou, x[iter1]) + np.dot(W_ou, s) )
    print(o)

    print("-- g --")
    g=np.tanh(  np.dot(U_g, x[iter1]) + np.dot(W_g, s) ) #wait, but why?
    print(g)
 
    print("-- c --")
    c=c*f + g*i #How come, that dimensions do not match?????????
    print(c)

    print("-- s NOW --")
    s=np.tanh(c)*o
    print(s)

    print("-- y aka. output NOW --")
    print("debug")
    lstm_output=np.dot(V,s)
    print(lstm_output)
    y_hat.append(softmax(lstm_output))
    print(y_hat[iter1])

print(" -------------- iteration ended ------------------")

