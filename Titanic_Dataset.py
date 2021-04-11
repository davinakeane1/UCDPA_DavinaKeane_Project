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

#bar chart to see survived vs not survived
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.bar(survived_figure.index, survived_figure.values, color='lavender',  edgecolor='purple')
plt.title('Survived vs Did Not survive')
plt.xticks([0,1],['Did Not survive', 'Survived'])
for i, value in enumerate(survived_figure.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
plt.show()

#bar chart to see survived by passenger gender
survived_gender = titanic_df.groupby('Sex')['Survived'].sum()
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.bar(survived_gender.index, survived_gender.values, color='lavender',  edgecolor='purple')
plt.title('Survived By Gender')
for i, value in enumerate(survived_gender.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
plt.show()