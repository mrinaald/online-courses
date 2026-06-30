#!/usr/bin/env python
# coding: utf-8

# # Practice Assignment 1 - Blackjack

# Welcome to Course 2 Practice Assignment 1. In this notebook you will see the optimal policy for Blackjack with different deal policies.
# 
# In a previous video (Solving the Blackjack Example) we described the optimal policy for Blackjack when the dealer sticks at 17. How would the optimal policy change if the dealer's policy were different? Would the optimal agent play more conservatively?
# 
# We ran experiments similar to Example 5.3 in the textbook but with different dealer policies. You can change dealer_sticks in the cell below to another number between 12 and 20 and see the optimal policy against that dealer.
# 
# This notebook is not graded. You do not need to write down any answers.

# In[4]:


# try modifying this variable for values between 12 and 20 to see the value of different policies
dealer_sticks = 13
# You don't really need to do anything here
# your code here


from IPython.display import Image 
assert 11<dealer_sticks<21, 'Please provide a number between 12 and 20.'
Image(filename='plots/plt_'+str(dealer_sticks)+'.png')


# In[2]:


print('run me to get your completion points :)')

