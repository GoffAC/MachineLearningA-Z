#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:11:06 2018

@author: alexandergoff
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

import math
N = 10000 #number of test customers
d = 10 #number of adverts

#we will also want to know which ads were selected by our system and hence need to store that value somewhere
ads_selected = []

#Lets make something to keep the scores (whichones were clicked in this single test) 
#of each advert
#This command makes and empty array, and then makes it have d number of columns
numbers_of_selections = [0] * d


#Sums of reward are another vector array to keep scores (before each test)
#We could have used the np.ones(d) notation as before
sums_of_rewards = [0] * d

#we also may want to sum the rewards across all to see how many times we were successful
total_reward = 0


#Now lets run this for every customer (N)
#the only issue though is at the beginning all are assumed to be the same
#as such the algorithm below will not be able to decide / may cause an issue
#as such for the first 10 tries, we will use a different selection method
for n in range(0, N):
    #as we are calculating the UCB for each of the 10 adverts each time
    #and then selecting the largest, we need to compare and then store 
    #the value from each round and the id of which ad it was
    ad = 0
    max_upper_bound = 0
    #now for each add we need to cal UCB Formula
    for i in range(0, d):
        #if this ad has been seen once, be normal - do UCB
        if (numbers_of_selections[i] > 0):
            #here we are trying to understand which of the ads have the highet upper bound
            #first lets calculate the average chance of reward for this add (at this n occasion)
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #next we need to calc delta which is a weighting factor for the number of times the advert
            #has been clicked compared to the number of occasions. As it is weighted it has a weird formula
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            #it is n+1 as the first value of n =0 but we need it to be equal to 1
            
            #now lets calculate the upper bound for this single advert option
            upper_bound = average_reward + delta_i
            
        else:
            #if this advert hasn't been seen yet, make sure it wins this round (give it a high number)
            upper_bound = 1e400
            #now if the UCB is larger than the previous maxUCB then we need to replace it
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i 
    #add one to the number of ads selected to go infront of the customer            
    ads_selected.append(ad)
    #add one to the correct column within the click_number            
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    #now we need to look at whether or not it worked
    #so we would need to check our prediction against the users action and determine if there should
    #be a reward
    #We will do this manually with the dataset provided which contains what the user would do if 
    #any of the ads were shown to them
    #obviously this isn't real data but will be applied here
    #1- Find our deserved reward (could be zero, could be 1, could be 10)
    reward = dataset.values[n, ad]
    #2- Add that reward to our sums of reward so we keep track of which are successful
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    #add the reward to the total reward to see our overall success rate
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()