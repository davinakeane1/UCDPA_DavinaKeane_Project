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

#Shows NaN values - Add a print statement to show
titanic_df.isnull().sum()

#See survived (1) vs not survived (0), grouped dataframe by Survived values and used count for number
survived_figure = titanic_df.groupby('Survived')['Survived'].count()
#print(survived_figure)

#bar chart to see survived vs not survived
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.bar(survived_figure.index, survived_figure.values, color='lavender',  edgecolor='purple')
plt.title('Survived vs Did Not survive')
plt.xticks([0,1],['Did Not survive', 'Survived'])
for i, value in enumerate(survived_figure.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
#plt.show()

#bar chart to see survived by the passenger genders
survived_gender = titanic_df.groupby('Sex')['Survived'].sum()
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.bar(survived_gender.index, survived_gender.values, color='lavender',  edgecolor='purple')
plt.title('Survived By Gender')
for i, value in enumerate(survived_gender.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
#plt.show()

#To see the passenger class (has 3 values), grouped dataframe by Pcalss values and used count for number
passenger_class_count = titanic_df.groupby('Pclass')['Pclass'].count()
#print(passenger_class_count)

#Pie chart to see the passenger class distribution
plt.figure()
colors = ['#ff6666', '#ffcc99', '#99ff99']
plt.title("Grouped by Passenger Class")
plt.pie(passenger_class_count.values, labels=["1st Class", "2nd Class", "3rd Class"], textprops={"fontsize":12}, autopct="%1.1f%%", colors = colors)
plt.tight_layout()
#plt.show()

#All not NaN values in Age stored to Numpy array called passenger_ages
passenger_ages = titanic_df[titanic_df['Age'].notnull()]['Age'].values
#print(passenger_ages.shape)

#histogram using histogram function from Numpy
ages_histogram = np.histogram(passenger_ages, bins=[0,10,20,30,40,50,60,70,80,90, 100])

#Created labels to make histogram more readable
ages_histogram_labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71–80", "81–90", "91-100"]

#histogram to see passenger age groups
plt.figure(figsize=(8,8))
plt.title('Passenger Age Groups')
plt.bar(ages_histogram_labels, ages_histogram[0], color='grey',  edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of passengers')
for i, bin in zip(ages_histogram[0], range(9)):
    plt.text(bin, i+3, str(int(i)), color='black', fontsize=10, style='oblique', horizontalalignment='center')
#plt.show()

#Passnger class and surivial rates
#print(titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Dropping the NaN from the Fare column
titanic_df = titanic_df.dropna(subset=['Fare'])

#Box plot of the fare distribution data
plt.figure(figsize=(14,8))
plt.title("Fare Range")
plt.boxplot(titanic_df["Fare"], vert=False)
#plt.show()

#Violin plot of the fare distribution data
plt.figure(figsize=(14,8))
plt.title("Fare Range")
plt.violinplot(titanic_df["Fare"], vert=False)
#plt.show()

#Seeing description of Fare column
#print(titanic_df["Fare"].describe())

#To see the unique values from Embarked column
#print(titanic_df['Embarked'].unique())

##To see the passenger embarked from (has 3 values), grouped dataframe by Embarked values and used count for number
passenger_embarked_count = titanic_df.groupby('Embarked')['Embarked'].count()
#print(passenger_embarked_count)

#bar chart to see passengers embarked from
plt.figure(figsize=(8,8))
plt.style.use('seaborn-darkgrid')
plt.bar(passenger_embarked_count.index, passenger_embarked_count.values, color='grey',  edgecolor='black')
plt.title('Passengers Embarked From')
for i, value in enumerate(passenger_embarked_count.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
#plt.show()

#Rows where the passengers embarked from Cherbourg
cherbourg_passengers = titanic_df.loc[titanic_df["Embarked"]=="C"]
#print(cherbourg_passengers)

#Rows where the passengers embarked from Queenstown
queens_passengers = titanic_df.loc[titanic_df["Embarked"]=="Q"]
#print(queens_passengers)

#Male Passengers embarking from Queenstown
age_queens_male = titanic_df.loc[(titanic_df["Sex"] == "male") & (titanic_df["Embarked"]=="Q")]
print(age_queens_male)

#Female Passengers embarking from Queenstown
age_queens_female = titanic_df.loc[(titanic_df["Sex"] == "female") & (titanic_df["Embarked"]=="Q")]
print(age_queens_female)