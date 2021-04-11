import pandas  as pd
import numpy as np
import matplotlib.pyplot as  plt
import os
import seaborn as sns

#showing working directory
os.getcwd()

#changing my working directory
os.chdir(r"C:\Users\College\Desktop\DataAnalytics")

#loading in the csv file
titanic_df = pd.read_csv("tested.csv", encoding = "ISO-8859-1")

#Shows NaN values - Add a print statement
titanic_df.isnull().sum()

#See survived (1) vs not survived (0)
survived_figure = titanic_df.groupby('Survived')['Survived'].count()
print(survived_figure)