# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing libraries
import pandas as pd
import os
import numpy as np
import math
import textwrap as tx
import matplotlib.pyplot as plt

#Setting directory
os.chdir('C:/Users/Ashtami/Documents/Python')

#Reading from csv data file
dataset = pd.read_csv('Dataset 3.csv',sep=',') #Reading from csv data file
print(dataset)
N = 10  #number of equations
P = 5 # number of columns
NPOP = 500 #number of population


#Dropping last two columns from dataset
dataset = dataset.drop(dataset.columns[14:16],axis=1)

#Dropping unwanted columns and keeping only first 5 columns
ds = dataset.drop(dataset.columns[5:13],axis=1)

#Creating predictor and target variables X and Y respectively
X = ds.iloc[:,:-1].values # :-1 specifies take all columns except last
Y = ds.iloc[:,-1].values #-1 specifies takes only the last column

#Splitting dataset into training and test data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0)

#Normalizing training dataset (input and output in the range 0 to 1)
from sklearn.preprocessing import Normalizer
sc_X = Normalizer()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_Y = Normalizer()
Y_train = sc_Y.fit_transform(Y_train)
Y_train= Y_train.reshape(-1,1)
Y_train = np.transpose(Y_train)

#==========================FUNCTIONS===========================================
#==============================================================================
#Function to create weight matrix of 10*5
#def funcWeight():
#    return np.random.uniform(low = -1, high = 1, size=(10,5))


#Function to normalize weight matrix
#Normalizing weight in the range 0-1, multiplying by 1000 and rounding it off to nearest integer
def funcNormalize(w):
    W = (w-w.min())/(w.max()-w.min())
    W_normal = np.round(W*1000).astype(int)
    return W_normal


#Function creating binary equivalent
def funcBin(x):
     binary_wt = np.empty((x.shape[0],x.shape[1])).astype(str) #create 2D empty array
     for i in range(x.shape[0]):         
         for j in range(x.shape[1]):
             binary_wt[i,j] = bin(x[i,j])[2:].zfill(10)
     return binary_wt


#function to transpose a matrix
def funcTranspose(x):
    return x.transpose()

#Function for matrix multiplication
def funcMatMul(XTrain, Wpn):
    return np.matmul(XTrain, Wpn)

#Equation 1
def funcX(x):
    a = [None]*len(x) #initializing empty array of size equal to length of Yhat(189)
    for i in range(len(x)):     #len(x) gives number of rows
        s = 0
        for j in range(x.shape[1]):     #x.shape[1] gives column count
            s += 1/(1 + math.exp(-x[i,j]))
        a[i] = s
    return a


#Equation 2: Function to compute fitness value
def funcFitness(Yhat, Y_train):
    s = 0
    for i in range(len(Yhat)):
        s += math.pow(Yhat[i]-Y_train[i],2)        
    return(1 - (s/len(Yhat))*100)


#Function to find fittest value
def funcFittest(Wpn, Y_train):
    fittest = [None]*len(Wpn)
    for i in range(len(Wpn)):
        X = funcMatMul(X_train, funcTranspose(np.array(Wpn[i])))
        YHAT = funcX(X)
        fittest[i] = funcFitness(YHAT, Y_train)
        #position = np.where(max(np.array(fittest)))
    return (fittest.index(max(fittest)), fittest)

#Function to return new chromosomes having higesh fitness value to become new set of chromosomes
def funcEliminator(Wpn, Y_train, mutants):
    topper = [None]*len(Wpn)
    newchromosome = [None]*int(len(mutants)/2)
    for i in range(len(Wpn)):
        X = funcMatMul(X_train, funcTranspose(np.array(Wpn[i])))
        YHAT = funcX(X)
        topper[i] = funcFitness(YHAT, Y_train)
        #position = np.where(max(np.array(fittest)))
        #topper = sorted(topper, reverse=True)
    topindex = np.argsort(topper)[::-1] #Reversing array to get index of fitness values sorted in descending order
    x = topindex[:int(len(mutants)/2)]
    for i in range(len(x)):
        newchromosome[i] = mutants[x[i]]
    return newchromosome


#Function to create entire population of weights in -1 to 1 range
def funcPopulation(n):
    return np.random.uniform(low = -1, high = 1, size=(n,N*P))

#Function to create 500 weight matrices from population
def funcWeightmatrix(x):
    W = [None]*x.shape[0]
    for i in range(x.shape[0]):
        A=x[i][0:50]
        A.resize((10, 5))
        W[i] = A
    return W
        

#Functin to create Npop (stores binary version of 500 weight matrices)
def funcNpop(population):
    npop = np.empty((population.shape[0],population.shape[1])).astype(str)    
    npop = funcBin(funcNormalize(population))
    return npop


#Functin to create chromosomes
def funcChromosome(x):
    Chromosome = [''] * x.shape[0]
    for i in range(500):  
        c = ''
        for j in range(50):
            c += x[i,j]
        Chromosome[i] = c
    return Chromosome


#Function for crossover
def funcCrossover(parent, chromosome):
     #Offspring = np.empty((500,2)).astype(str) 
     Offspring = []
     crossoverPoint = np.random.randint(0,len(parent))
     print(crossoverPoint)
     for i in range(len(chromosome)):
         leftParent = parent[0:crossoverPoint]
         rightParent = parent[crossoverPoint:]
         leftChromosome = chromosome[i][0:crossoverPoint]
         rightChromosome = chromosome[i][crossoverPoint:]
         Offspring.append(leftParent+rightChromosome)
         Offspring.append(leftChromosome+rightParent)
    
     #x = np.reshape(Offspring, (500, 2))
     return(Offspring)
 
    
#Function to create mutants by taking 5% of total bits, Generating those many random numbers between 0-499
#Flipping bits at those random positions
def funcMutate(offspring):
    #assuming 5% mutation
    #Mutant = Offspring
    n = np.round(0.05*len(offspring)).astype(int)
    randombit = np.random.randint(low = 0, high = 500, size=(n))
    #print(randombit[0],randombit[1], randombit[2],randombit[3], randombit[4])
    for i in range(len(offspring)):
        for j in range(n):
            if offspring[i][randombit[j]] == '0':
                offspring[i] = list(offspring[i])
                offspring[i][randombit[j]] = '1'
                offspring[i] = ''.join(offspring[i])
            else:
                offspring[i] = list(offspring[i])
                offspring[i][randombit[j]] = '0'
                offspring[i] = ''.join(offspring[i])
    return offspring




#Function to debinarize         
def funcDebinarize(mutant):
    debinarize = np.empty((len(mutant),N*P))
    for i in range(len(mutant)):
        t = tx.wrap(mutant[i],10)
        for j in range(N*P):
            d = int(t[j],2)                    #int('string',2) is converting to decimal
            d/=1000         
            debinarize[i][j] = d
    return debinarize
 
#Function to denoramlize
def funcDenormalize(debinarize):
    denormalize = np.empty((debinarize.shape[0],debinarize.shape[1]))
    Max = max(max(x) for x in denormalize)
    Min = min(min(x) for x in denormalize)
    
    for i in range(debinarize.shape[0]):
        for j in range(debinarize.shape[1]):
            denormalize[i][j] = (2*(debinarize[i][j]- Min/ Max - Min))-1
    return denormalize



#==============================================================================
#==============================================================================


#Calling function to create population 
population = funcPopulation(500)

#Calling function to create Npop which is a matrix of 500*50 having binary values
Npop = funcNpop(population)

#Calling function to create 500 Chromosomes
Chromosome = funcChromosome(Npop)

#Calling function to create 500 weight matrices
WeightMatrix = funcWeightmatrix(population)

#Calling function to find position of fittest value
Fittest = funcFittest(WeightMatrix, Y_train)    #Stores position of max fitness as well as fitness values

#Parent Chromosome
Parent = Chromosome[Fittest[0]]

# Simulate all of the generations.

i = 0
fittestval = list()
parentchromosome = list()
#PotentialParent = ''
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

fittestval.append(max(Fittest[1]))
parentchromosome.append(Parent)

while i < 10:
    
    print("Iteration: " , i)  
     
    Offspring = funcCrossover(Parent, Chromosome)           #Creating offspring by crossover
    Mutants = funcMutate(Offspring)                         #Creating Mutants
    Debinarize = funcDebinarize(Mutants)                    #Calling function to perform Debinarization
    Denormalize = funcDenormalize(Debinarize)               #Calling a function to denormalize  
    WeightMatrix = funcWeightmatrix(Denormalize)            #Calling a function to create new weight matrix having values between -1 to 1
    Chromosome = funcEliminator(WeightMatrix, Y_train, Mutants) #Calling a function to get list of max 500 values from 1000 
    Fittest = funcFittest(WeightMatrix, Y_train)
    fittestval.append(max(Fittest[1])) 
    PotentialParent = Chromosome[Fittest[0]] 
    Chromosome = funcEliminator(WeightMatrix, Y_train, Mutants) #Reducing size of Npop from 1000 to 500
    
    print("Fitness Value : ", max(Fittest[1]))
    
    if max(Fittest[1]) < fittestval[-1]:
        Parent = parentchromosome[-1]
    else:
        Parent = PotentialParent
        parentchromosome.append(Parent)
        
    plt.scatter(i, fittestval[-1])
    i += 1
    plt.show()


#Scatter Plot
from mpl_toolkits.mplot3d import Axes3D as fig
x1=Input[['Weight lbs']]
x2=Input[[ 'Height inch']]

plt.scatter(x1, x2,yhat)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('yhat')

plt.show()

#MSE
d2=(yhat-Output_test)**2
ydif=d2.sum(axis=0)
Overall_error=ydif/len(Output_test)








