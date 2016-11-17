import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from random import normalvariate

corruption = 0.5
def create_list(n,length,term,randomize=False):
	list = []
	for _ in range(n):
		list.append([term(x) for x in range(length)])
		if randomize == True:
			list = [map(randomx,e) for e in list]
	return list
'''
n 	 -- number of elements in the list
term -- a function that takes one argument
'''
sinx = lambda x: sin(x/2)
cosx = lambda x: cos(x/2)

randomx = lambda x: normalvariate(0,corruption)+x

def uptrend(term):
	return lambda k: term(k)+(0.1*k)

def downtrend(term):
	pass

def straight():
	lambda x: 0.5
'''
list = create_list(1,100,uptrend(sinx),True)

#list = create_list(1,100,sinx,True)
print list[0][:10]

index = range(len(list[0]))
plt.plot(index, list[0])
plt.show()
'''