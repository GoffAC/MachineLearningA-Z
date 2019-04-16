#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:11:06 2018

@author: alexandergoff
"""
# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#First lets make something to keep the scores (whichones were clicked in this single test) 
#of each advert
#This command makes and empty array, and then makes it have d number of columns
d = 10
click_number = [0] * d

#Sums of reward are another vector array to keep scores (before each test)
#We could have used the np.ones(d) notation as before
sumsOfReward = [0] * d

#we also may want to sum the rewards across all to see how many times we were successful
totalReward = 0

#we will also want to know which ads were selected by our system and hence need to store that value somewhere
adsSelected = []

#Now lets run this for every customer (N)
#the only issue though is at the beginning all are assumed to be the same
#as such the algorithm below will not be able to decide / may cause an issue
#as such for the first 10 tries, we will use a different selection method

N = 10000

for n in range(0, N):
    
    #as we are calculating the UCB for each of the 10 adverts each time
    #and then selecting the largest, we need to compare and then store 
    #the value from each round and the id of which ad it was
    maxUCB = 0
    ad = 0
    
    #now for each add we need to cal UCB Formula
    for i in range(0, d):
        
        #if this ad has been seen once, be normal - do UCB
        if (click_number[i] > 0):
            #here we are trying to understand which of the ads have the highet upper bound
            #first lets calculate the average chance of reward for this add (at this n occasion)
            averageReward = sumsOfReward[i] / click_number[i]
            #next we need to calc delta which is a weighting factor for the number of times the advert
            #has been clicked compared to the number of occasions. As it is weighted it has a weird formula
            delta_i = math.sqrt(3/2 * math.log(n + 1) / click_number[i])
            #it is n+1 as the first value of n =0 but we need it to be equal to 1
            
            #now lets calculate the upper bound for this single advert option
            UCB = averageReward + delta_i
            
        else:
            #if this advert hasn't been seen yet, make sure it wins this round (give it a high number)
            UCB = 1e400
            
            #now if the UCB is larger than the previous maxUCB then we need to replace it
        if UCB > maxUCB:
            maxUCB = UCB
            ad = i 
    #add one to the number of ads selected to go infront of the customer            
    adsSelected.append(ad)
    #add one to the correct column within the click_number            
    click_number[ad] = click_number[ad] + 1
    #now we need to look at whether or not it worked
    #so we would need to check our prediction against the users action and determine if there should
    #be a reward
    #We will do this manually with the dataset provided which contains what the user would do if 
    #any of the ads were shown to them
    #obviously this isn't real data but will be applied here
    #1- Find our deserved reward (could be zero, could be 1, could be 10)
    reward = dataset.values[n, ad]
    #2- Add that reward to our sums of reward so we keep track of which are successful
    sumsOfReward[ad] = sumsOfReward[ad] + reward
    #add the reward to the total reward to see our overall success rate
    totalReward = totalReward + reward
    
print ("fin")
print (totalReward)