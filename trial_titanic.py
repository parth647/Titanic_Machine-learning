# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 19:49:10 2018

@author: parth
"""
import pandas as pd 
import seaborn as sns  # for statistical analysis

titanic_df = pd.read_csv("train.csv") #fetch the data from the train file
titanic_df=titanic_df.fillna(titanic_df.median())#fill the data with missing values
#remove the unecessary data
titanic_df['family']=titanic_df['SibSp']+titanic_df['Parch']
titanic_df=titanic_df.drop(['PassengerId','Name','Ticket','SibSp','Parch'], axis=1)
  #lets do some analysis
  
#How many males and females survived 
sns.countplot(x='Survived',hue='Sex',data=titanic_df,order=[1,0])
#age wise probablity of people surviving
#You should know how facetGrid works
facet = sns.FacetGrid(titanic_df, hue="Survived",aspect=5)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, titanic_df['Age'].max()))
facet.add_legend()