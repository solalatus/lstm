import numpy as np


def sigmoid(x):                                        
    return 1 / (1 + np.exp(-x))


word_dim = 5
hidden_dim = 2
#bptt_truncate = bptt_truncate

vocab=np.arange(word_dim)

U_in = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_fo = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_ou = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
U_g = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))


#V_in = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
#V_fo = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
#V_ou = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))

W_in = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_fo = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_ou = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
W_g = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))


x=np.array([[0,0,0,0,1],[1,0,0,0,0],[0,0,1,0,0]])
o=np.array([[0,0,0,0,0],[1,0,0,0,0],[0,0,1,0,0]])

h=np.array([[0,0],[0,0],[0,0]])

print "weight U: ", U_in
print "weight W: ", W_in
print "---------------init done-------------"
for i in range(len(x)):
    #--------
    print "input layer calculation"
    print "input: ", x[i], " "

    print "np.dot(U_in, x[i])  ", np.dot(U_in, x[i])
    print " np.dot(W_in, h[i]) ",  np.dot(W_in, h[i]) 
    inpu = sigmoid( np.dot(U_in, x[i]) + np.dot(W_in, h[i]) )
    print "-------"
    print i
    print "---------------------------------------"

    forget = sigmoid( np.dot(U_fo, x[i]) + np.dot(W_fo, h[i]) )
    out = sigmoid( np.dot(U_ou, x[i]) + np.dot(W_ou, h[i]) )
    g=np.tanh(  np.dot(U_g, x[i]) + np.dot(W_g, h[i]) ) #wait, but why?
    
    if i == 0:
	hi=np.array([0,0,0,0,0])
    else:
        hi=o[i-1]
    
    o[i]=hi*forget + g*inpu #How come, that dimensions do not match?????????
    h[i]=np.tanh(o[i])*out

print  " -------------- iteration ended ------------------"
print "hiddens: ", h
print "outputs: ", o
