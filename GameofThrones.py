# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 00:14:58 2023

@author: Yunus

Data Science Projects @ Great Learning

EDA on Game of Thrones Data
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# data import
battle = pd.read_csv('battles.csv')

# first five rows of data
print (battle.head())

# number of columns and rows of data
print (battle.shape)

# changing column name, inplace = True means the data is modified in place
battle.rename(columns={'attacker_1':'primary_attacker'},inplace=True)
battle.head()

# changing column name
battle.rename(columns={'defender_1':'primary_defender'},inplace=True)
battle.head()

# number of attacker_king 
battle['attacker_king'].value_counts()

# number of location
battle['location'].value_counts()

# graph of attacker size of attacker kings
sns.set(rc={'figure.figsize':(13,5)})
sns.barplot(x='attacker_king',y='attacker_size',data=battle)
plt.show()

# graph of defender size of defender kings
sns.set(rc={'figure.figsize':(13,5)})
sns.barplot(x='defender_king',y='defender_size',data=battle)
plt.show()

# battle types of attacker kings
sns.countplot(x=battle['attacker_king'],hue=battle['battle_type'])
plt.show()

# data import
death = pd.read_csv('character-deaths.csv')
death.head()

# number of columns and rows of data
death.shape

# number of gender of deaths
death['Gender'].value_counts()

# number of nobility of deaths
death['Nobility'].value_counts()

# graph of death year
sns.countplot(data = death, x ='Death Year')
plt.show()

# graph of Allegiances
sns.set(rc={'figure.figsize':(30,10)})
sns.countplot(x=death['Allegiances'])
plt.show()
