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
titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

# plot
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=4)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)