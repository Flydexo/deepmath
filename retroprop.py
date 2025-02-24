import numpy as np
import matplotlib.pyplot as plt
import random

def sigma(x):
    return 1/(1+np.exp(-x))

carres_rouges = [(1,1), (2,0.5), (2,2), (3,1.5), (3,2.75), (4,1), (4,2.5), (4.5,3), (5,1), (5,2.25)]
ronds_bleus   = [(0,3), (1,1.5), (1,4), (1.5,2.5), (2,2.5), (3,3.5), (3.5,3.25), (4,3), (4,4), (5,4)] 
N = len(carres_rouges) + len(ronds_bleus)

def F(a,b,c,x,y): 
    return a*x+b*y+c

def error(a,b,c):
    e_rouge = sum([(sigma(F(a,b,c,x,y)) - 1)**2 for (x,y) in carres_rouges])
    e_bleu = sum([(sigma(F(a,b,c,x,y)))**2 for (x,y) in ronds_bleus])
    return (e_rouge+e_bleu)/N

def grad(a,b,c):
    delta_a_rouge = sum([2*x*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for (x,y) in carres_rouges]) 
    delta_a_bleu = sum([2*x*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*sigma(a*x+b*y+c) for (x,y) in ronds_bleus]) 

    delta_b_rouge = sum([2*y*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for (x,y) in carres_rouges])
    delta_b_bleu = sum([2*y*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)) for (x,y) in ronds_bleus])

    delta_c_rouge = sum([2*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)-1) for (x,y) in carres_rouges])
    delta_c_bleu = sum([2*sigma(a*x+b*y+c)*(1-sigma(a*x+b*y+c))*(sigma(a*x+b*y+c)) for (x,y) in ronds_bleus])

    return 1/N*np.array([delta_a_bleu+delta_a_rouge, delta_b_bleu+delta_b_rouge, delta_c_bleu+delta_c_rouge])

def retro(p0):
    delta = 1
    pi = p0
    for _ in range(1000):
        pi = pi - delta*grad(pi[0], pi[1], pi[2])
        print(pi)
    return pi

def descente(grad_f, X0, delta=0.1, nmax=10):
    liste_X = [X0]
    liste_grad = []
    X = X0
    for i in range(nmax):
        gradient = grad_f(*X)
        X = X - delta*gradient
        liste_X.append(X)
        liste_grad.append(gradient)
    return liste_X, liste_grad

for x, y in carres_rouges:    # points
    plt.scatter(x, y, marker='s', color='red')
for x, y in ronds_bleus:    # points
    plt.scatter(x, y, color='blue')

X0 = np.array([random.random()*6, random.random()*6, random.random()*6])

[a,b,c] = retro(X0)
ap,bp,cp = descente(grad, X0, 1, 1000)[0][-1]

X = np.linspace(0,6,num=100)
Y = -1/b*(a*X+c)
Y2 = -1/bp*(ap*X+cp)

plt.plot(X,Y,color='red')
plt.plot(X,Y2,color='green', linestyle='dashed')
plt.show()