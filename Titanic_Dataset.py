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
plt.show()

#bar chart to see survived by the passenger genders
survived_gender = titanic_df.groupby('Sex')['Survived'].sum()
plt.figure()
plt.style.use('seaborn-darkgrid')
plt.bar(survived_gender.index, survived_gender.values, color='lavender',  edgecolor='purple')
plt.title('Survived By Gender')
for i, value in enumerate(survived_gender.values):
    plt.text(i, value-60, str(value), color='black', fontsize=10, style='oblique', horizontalalignment='center')
plt.show()

#To see the passenger class (has 3 values), grouped dataframe by Pcalss values and used count for number
passenger_class_count = titanic_df.groupby('Pclass')['Pclass'].count()
#print(passenger_class_count)

#Pie chart to see the passenger class distribution
plt.figure()
colors = ['#ff6666', '#ffcc99', '#99ff99']
plt.title("Grouped by Passenger Class")
plt.pie(passenger_class_count.values, labels=["1st Class", "2nd Class", "3rd Class"], textprops={"fontsize":12}, autopct="%1.1f%%", colors = colors)
plt.tight_layout()
plt.show()

#All not NaN values in Age stored to Numpy array called passenger_ages
passenger_ages = titanic_df[titanic_df['Age'].notnull()]['Age'].values
#print(passenger_ages.shape)

#histogram using histogram function from Numpy
ages_histogram = np.histogram(passenger_ages, bins=[0,10,20,30,40,50,60,70,80,90, 100])

#Created labels to make histogram more readable
ages_histogram_labels = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71–80", "81–90", "91-100"]

#histogram to see passenger age groups
plt.figure(figsize=(8,8))
plt.title("Passenger Age Groups")
plt.bar(ages_histogram_labels, ages_histogram[0], color="grey",  edgecolor="black")
plt.xlabel("Age")
plt.ylabel("Number of passengers")
for i, bin in zip(ages_histogram[0], range(9)):
    plt.text(bin, i+3, str(int(i)), color="black", fontsize=10, style="oblique", horizontalalignment="center")
plt.show()


#Passnger class and surivial rates
pclass_survival_mean = titanic_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#using round to round  off figure to 2 decimal places
print(round(pclass_survival_mean, 2))

#Catplot to see passnger class and surivial rates
g = sns.catplot(x="Survived", y="Survived", hue="Pclass", col="Survived", data=pclass_survival_mean, kind="bar");
plt.show()

#Catplot to see passnger class and surivial rates using pclass_survival_mean
sns.catplot(x="Survived",data=pclass_survival_mean,kind="count",hue="Pclass")
plt.show()

#Catplot to see passnger class and surivial rates using whole dataframe
sns.catplot(x="Survived",data=titanic_df,kind="count",hue="Pclass")
plt.show()

#Dropping the NaN from the Fare column
titanic_df = titanic_df.dropna(subset=['Fare'])

#Box plot of the fare distribution data
plt.figure(figsize=(14,8))
plt.title("Fare Range")
plt.boxplot(titanic_df["Fare"], vert=False)
plt.show()

#Violin plot of the fare distribution data
plt.figure(figsize=(14,8))
plt.title("Fare Range")
plt.violinplot(titanic_df["Fare"], vert=False)
plt.show()

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
plt.show()

#Rows where the passengers embarked from Cherbourg
cherbourg_passengers = titanic_df.loc[titanic_df["Embarked"]=="C"]
#print(cherbourg_passengers)

#Rows where the passengers embarked from Queenstown
queens_passengers = titanic_df.loc[titanic_df["Embarked"]=="Q"]
#print(queens_passengers)

#Male Passengers embarking from Queenstown
age_queens_male = titanic_df.loc[(titanic_df["Sex"] == "male") & (titanic_df["Embarked"]=="Q")]
#print(age_queens_male)

#Female Passengers embarking from Queenstown
age_queens_female = titanic_df.loc[(titanic_df["Sex"] == "female") & (titanic_df["Embarked"]=="Q")]
#print(age_queens_female)

#Passengers embarking from Queenstown class 1 & 2 breakdown
queens_class_1_2 = titanic_df.loc[(titanic_df["Pclass"] <= 2) & (titanic_df["Embarked"]=="Q")]
#print(queens_class_1_2)

#Passengers embarking from Queenstown class 3 breakdown
queens_class_3 = titanic_df.loc[(titanic_df["Pclass"] == 3) & (titanic_df["Embarked"]=="Q")]
#print(queens_class_3)

#Class of the females embarking from Queenstown
queens_female_class = titanic_df.loc[(titanic_df["Sex"] == "female") & (titanic_df["Embarked"]=="Q") & (titanic_df["Pclass"] >= 3)]
#print(queens_female_class)

#Class of the males embarking from Queenstown
queens_male_class = titanic_df.loc[(titanic_df["Sex"] == "male") & (titanic_df["Embarked"]=="Q") & (titanic_df["Pclass"] >= 1)]
#print(queens_male_class)

#First class of the females embarking from Queenstown
queens_female_class_1 = titanic_df.loc[(titanic_df["Sex"] == "female") & (titanic_df["Embarked"]=="Q") & (titanic_df["Pclass"] == 1)]
#print(queens_female_class_1)

#Using .iterrows() to print out each passengers namd and what class the were in
#for index, row in titanic_df.iterrows():
    #print(row[3], 'was in class', row[2])

#shows the unique values in the dataframe
#print(titanic_df.nunique())

#Heatmap to show missing values in dataset - before inputting missing age values
ax = plt.axes()
sns.heatmap(titanic_df.isnull(), cbar=True)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
ax.set_title('NaN Values in Dataset')
plt.show()

#Filling in the missing values in the Aged column using the median value
passenger_ages_withmedian = titanic_df["Age"] = titanic_df["Age"].fillna((titanic_df["Age"].median()))

#Checking the above worked
#print(titanic_df["Age"].isnull().sum())

#histogram using histogram function from Numpy - with median value filled in for NaN
ages_histogram_withmedian = np.histogram(passenger_ages_withmedian, bins=[0,10,20,30,40,50,60,70,80,90, 100])

#Created labels to make histogram more readable - with median value filled in for NaN in data
ages_histogram_labels_withmedian = ["0–10", "11–20", "21–30", "31–40", "41–50", "51–60", "61–70", "71–80", "81–90", "91-100"]

#histogram to see passenger age groups - with median value filled in for NaN
plt.figure(figsize=(8,8))
plt.title("Passenger Age Groups- using median for NaN")
plt.bar(ages_histogram_labels_withmedian, ages_histogram_withmedian[0], color="black",  edgecolor="grey")
plt.xlabel("Age")
plt.ylabel("Number of passengers - using median for NaN")
for i, bin in zip(ages_histogram_withmedian[0], range(9)):
    plt.text(bin, i+3, str(int(i)), color="grey", fontsize=10, style="oblique", horizontalalignment="center")
plt.show()

#Catplot boxplot to show the average ages of the passengers in each class
AverageAgePerClass = sns.catplot(data = titanic_df , x = 'Pclass' , y = 'Age', kind = 'box', palette="Set3")
sns.set_theme(style="ticks")
plt.show()

#Getting Pclass 1 mean
titanic_df[titanic_df['Pclass'] == 1]['Age'].mean()

#Getting Pclass 2 mean
titanic_df[titanic_df['Pclass'] == 2]['Age'].mean()

#Getting Pclass 3 mean
titanic_df[titanic_df['Pclass'] == 3]['Age'].mean()

# CUSTOM FUNCTION - Imputation for missing age variables
def impute_null_age_vaues(cols):
    passenger_age = cols[0]
    passenger_c = cols[1]

    if pd.isnull(passenger_age):
        if (passenger_c == 1):
            return titanic_df[titanic_df['Pclass'] == 1]['Age'].mean()
        elif (passenger_c == 2):
            return titanic_df[titanic_df['Pclass'] == 2]['Age'].mean()
        elif (passenger_c == 3):
            return titanic_df[titanic_df["Pclass"] == 3]["Age"].mean()
    else:
        return passenger_age

#Applying custom function to the dataset
titanic_df["Age"] = titanic_df[["Age", "Pclass"]].apply(impute_null_age_vaues, axis = 1)

#Heatmap to show missing values in dataset - AFTER inputting missing age values
ax = plt.axes()
sns.heatmap(titanic_df.isnull(), cbar=True,  cmap="YlGnBu")
plt.xticks(rotation=30)
plt.yticks(rotation=30)
ax.set_title("No NaN values in Age Column")
plt.show()

#checking the NaN values in Cabin column
#print(titanic_df["Cabin"].isnull().sum())
#print(titanic_df["Cabin"].shape)

#Checking the shape of the data
#print(titanic_df.shape)

#Droppping the Cabin column from the dataframe
titanic_df.drop(["Cabin"], axis="columns", inplace=True)

#Checking the shape of the data
print(titanic_df.shape)