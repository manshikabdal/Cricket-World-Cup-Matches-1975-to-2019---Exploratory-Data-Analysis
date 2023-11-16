#!/usr/bin/env python
# coding: utf-8

# ## Cricket World Cup Matches 1975 to 2019 - Exploratory Data Analysis

# In[1]:


#importing all the nesscessary libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns


# In[2]:


# loading the dataset of cricket-world-cup from 1975 to 2019
cwc = pd.read_csv('CWC-1975-2019.csv', encoding = 'unicode_escape')


# In[3]:


#size of the dataframe
cwc.shape


# In[4]:


cwc.head()


# In[5]:


cwc.info()


# In[6]:


# Removming the unnesscary columns
columns_to_delete = ['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11']

cwc = cwc.drop(columns=columns_to_delete)

cwc.head(2)


# In[7]:


#Renaming the columns for better understanding
cwc = cwc.rename(columns = {'venue':'year', 'team1':'team_bat_first', 'score1':'score_first', 
                            'team2':'team_bat_second', 'score2':'score_second'})
cwc.head(1)


# In[8]:


# checking the null values 
cwc.isnull().sum()


# In[9]:


# getting the names of those columns which contains null values
[features for features in cwc.columns if cwc[features].isnull().sum() > 0]


# In[10]:


# heatmap for null values
sns.heatmap(cwc.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.title('Heat map of Cricket world cup')
plt.show()


# In[11]:


# spliting the runs scored from wicket out of both the teams
cwc['score_first'] = cwc['score_first'].str.split('/').str[0]
cwc['score_second'] = cwc['score_second'].str.split('/').str[0]


# In[12]:


cwc.head(2)


# In[13]:


# changing data type of score_first  
cwc['score_first'] = pd.to_numeric(cwc['score_first'])


# In[14]:


# checking what's the error at 96th row to rectify it
cwc.iloc[96,:]


# In[15]:


# replacing inappropriate value with 0
cwc.replace('24-Jan', 0, inplace= True)


# In[16]:


# changing data type of score_second 
cwc['score_second'] = pd.to_numeric(cwc['score_second'])


# In[17]:


cwc.dtypes


# In[18]:


# Calculating the mean of run scored by first team bated
mean_value1 = cwc['score_first'][cwc['score_first'] != ''].mean()

print(round(mean_value1))
##
# Calculating the mean of run scored by second team bated
mean_value2 = cwc['score_second'][cwc['score_second'] != ''].mean()

print(round(mean_value2))


# In[19]:


# replacing the empty value in score first with mean value
cwc['score_first'].replace(0 , mean_value1 ,inplace = True)

# replacing the empty value in score second with mean value
cwc['score_second'].replace(0, mean_value2 ,inplace = True)


# In[20]:


# extracting which team won the match
def extract_team_won(row):
    score_first = row['score_first']
    score_second = row['score_second']
    
    if score_first > score_second:
        return row['team_bat_first']
    elif score_first < score_second:
        return row['team_bat_second']
    else:
        return 'Draw'

            
cwc['team_won'] = cwc.apply(extract_team_won, axis=1)


# In[21]:


cwc


# In[22]:


country_names=cwc.team_won.value_counts().index
country_names


# In[23]:


cwc.isnull().sum()


# In[24]:


cwc1 = cwc[cwc.isna().any(axis=1)]
print (cwc1)


# In[25]:


# droping the null value 
cwc.dropna(inplace = True)


# In[26]:


cwc.isnull().sum()


# In[27]:


# droping the result column
cwc = cwc.drop(columns = 'result')


# In[28]:


cwc.head(1)


# In[29]:


# extracting the years from the venue column
cwc['Years'] = cwc['year'].str.extract(r'(\b\d{4}\b)')
cwc['Years']


# In[30]:


# Year and Years columns are same
cwc['year']=cwc['Years']

cwc.head(2)


# In[31]:


# Deleting the 'Years' column from the dataset
columns_to_delete2 = ['Years']
cwc = cwc.drop(columns = columns_to_delete2)
cwc.head(2)


# In[32]:


cwc.dtypes


# In[33]:


#Descriptive statistics
cwc.describe()


# ## Exploring the data about categorical Variables

# In[34]:


# frequency table for team won
frequency_table = cwc['team_won'].value_counts()
print(f'Frequency Table for team won :\n{frequency_table}\n')


# In[35]:


# pie chart showing Top five countries who had win the matches
matplotlib.rcParams['figure.figsize'] = (12, 6)
plt.pie(frequency_table[:5],labels=frequency_table.index[:5],autopct='%1.2f%%')
plt.title('Pie Chart of Team Won')
plt.show()


# In[36]:


# bar plot for how many matches won by teams in cricket world cup.
matplotlib.rcParams['figure.figsize'] = (12, 6)
sns.barplot(x=frequency_table , y=frequency_table.index ,data=cwc,orient='horizontal' ,palette='viridis')
plt.title('Bar Plot of Team Won')
plt.xlabel('Team')
plt.ylabel('Count')
plt.show()


# ## Exploring the data about Numerical Variables

# In[37]:


# making a list of score_first and score_second 
teams_scores = ['score_first', 'score_second']

# histogram to visualize the distribution of teams_scores
teams_scores = cwc.select_dtypes(include=['float', 'float']).columns
for column in teams_scores:
    plt.hist(cwc[column], bins=10, color='blue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel('scores')
    plt.ylabel('Frequency')
    plt.show()


# In[38]:


info1 = cwc['score_first'].describe()
print(info1)
info2 = cwc['score_second'].describe()
print(info2)


# In[39]:


# box plot for identify central tendencyand spread of teams_scores
for column in teams_scores:
    sns.boxplot(x=cwc[column])
    plt.title(f'Box Plot of {column}')
    plt.show()


# In[42]:


# pair plot for relationship between teams_scores
matplotlib.rcParams['figure.figsize'] = (12, 6)
sns.pairplot(cwc[teams_scores])
plt.show()


# In[43]:


# regplot of score_first and score_second
sns.regplot(cwc['score_first'],cwc['score_second'])
plt.title('Reg plot of score_first and score_second')
plt.show()


# In[44]:


#correlation matrix to show correlation between score_first and score_second
correlation_matrix = cwc[teams_scores].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[45]:


cwc.corr()


# # Conclusion
# 1. Score_first histogram is symmetric, this means the mean and median are close together.
#    Score_second histogram is right skewed, so the mean is larger than the median.
# 2. Minimum run scored by a team who batted first is 36 and the team who batted second scored 32.
#    Maximum run scored by a team who batted first is 417 and the team who batted second scored 338.
# 3. Box plot represents all the central tendency values of the team who batted first and second.
# 4. The positive slope of rep plot represents that there is the positive linear relation between score_first and score_second.
#    As the value of independent variable, i.e. score_first increase, the values of dependent variables, i.e. score_second
#    also increases.
# 5. There a is +ve correlation between score_first and score_second.

# In[ ]:





# In[ ]:




