#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing Dataset
df = pd.read_csv(r'C:\Users\pv437\Desktop\Data Scince Folder\Projects\Project 1\bankruptcy-prevention.csv', delimiter=';')
print('Shape of the data',df.shape)
df.tail(10)


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.rename(columns={' management_risk':'management_risk',' financial_flexibility':'financial_flexibility',' credibility':'credibility',' competitiveness':'competitiveness',' operating_risk':'operating_risk',' class':'class'},inplace=True)


# In[7]:


df['class'].value_counts().get('bankruptcy')


# In[8]:


df['class'].value_counts().get('non-bankruptcy')


# In[9]:


#Creating Df1 To change Bankruptcy and Non- Bankruptcy into Integer
df1=df.copy()
maping={'bankruptcy':0,'non-bankruptcy':1}
df1['class']=df1['class'].map(maping)
print('After Changing Bankruptcy to 0 & Non-Bankruptcy to 1 we get our dataframe as ' )
df1


# In[10]:


#outlier ppt
ot=df1.copy() 
fig, axes=plt.subplots(7,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='industrial_risk',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='management_risk',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='financial_flexibility',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='credibility',data=ot,palette='crest',ax=axes[3])
sns.boxplot(x='competitiveness',data=ot,palette='crest',ax=axes[4])
sns.boxplot(x='operating_risk',data=ot,palette='crest',ax=axes[5])
sns.boxplot(x='class',data=ot,palette='crest',ax=axes[6])
plt.tight_layout(pad=2.0)


# The Above Boxplot shows  that our data has no outlier.Each feature has three unique values 0 , 0.5 , 1 Which signifies there respected significance

# In[11]:


# Creating  boxplot for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='class', y=feature, data=df,palette='rocket')
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)

plt.tight_layout()
plt.show()


# In[12]:


plt.figure(figsize=(15,12))
for i, predictor in enumerate(df.drop(columns=['class'])):
    ax=plt.subplot(3,2,i+1)
    sns.countplot(data=df,x=predictor,hue='class',palette='Set1')


# In[13]:


data=df1.copy()
correlations = data.corrwith(df1['class'])
correlations = correlations[correlations!=1]
positive_correlations = correlations[correlations >0].sort_values(ascending = False)
negative_correlations =correlations[correlations<0].sort_values(ascending = False)

correlations.plot.bar(
        figsize = (18, 10), 
        fontsize = 15, 
        color = 'orange',
        rot = 0, grid = True)
plt.title('Correlation with Class \n',
horizontalalignment="center", fontstyle = "normal", 
fontsize = "22", fontfamily = "sans-serif")


# In[14]:


# Feature Relationships
correlation_matrix = df1.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")


# In[15]:


# Count the values in the 'class' column
class_counts = df['class'].value_counts()
# Plot a basic pie chart
plt.pie(class_counts, labels=class_counts.index,
        autopct='%.2f%%', 
        explode = [0.04,0.03],
        shadow= True,
        textprops = {'size':'large','fontweight':'bold','rotation':'0','color':'black'},
        startangle=180 )
# Add a title
plt.title("Class Type Distribution")
# Show the pie chart
plt.show()


# In[16]:


df.groupby(['class']).count()


# In[17]:


sns.pairplot(df1, hue='class')


# In[18]:


x=df1.drop('class',axis=1)
x.head(5)


# In[19]:


y=df1['class']
y.head(5)


# In[20]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[21]:


# applying SelectKBest class to extract top 6 best features
bestfeatures = SelectKBest(score_func=chi2, k=6)
fit = bestfeatures.fit(x, y)


# High chi2 value suggest, feature is useful in predicting the class variable

# In[22]:


featureScores_univ = pd.DataFrame({'variables':x.columns, 'Score':fit.scores_})
featureScores_univ.sort_values(by=['Score'], ascending=False)


# In[23]:


from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import DecisionTreeClassifier
import matplotlib.pyplot as plt


# In[24]:


# use inbuilt class feature_importances of tree based classifiers
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)

# plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(6).plot(kind='barh', color='b')
plt.show()


# In[25]:


featureScores_dt = pd.DataFrame({'variables':x.columns, 'Score':model.feature_importances_})
featureScores_dt.sort_values(by=['Score'], ascending=False)


# In[ ]:





# In[26]:


#Splitting Dataset Into Training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[27]:


#Defining Decision Tree Model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)
model.fit(x_train, y_train)


# In[28]:


#Plotting the Decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(model,filled=True)
plt.show()


# In[29]:


#Creating pred Variable to store Predicted Values
pred=model.predict(x_test)


# In[30]:


#Creating New Dataset to Compare Predicted and Actual values
data_=pd.DataFrame({'Actual values':y_test,'Predicted Values':pred})
data_


# In[31]:


#Plotting Crosstab To check For True/False Positive/Negative Values 
pd.crosstab(y_test,pred)


# In[32]:


#Plotting Confusion Matrix and Classifiction report to check accuracy
from sklearn.metrics import confusion_matrix,classification_report
sns.heatmap(confusion_matrix(y_test,pred),annot=True,fmt = "d",linecolor="k",linewidths=3)
print(classification_report(y_test,pred))


# In[33]:


#Testing Score
Test_score=model.score(x_test,y_test)
Test_score


# In[34]:


#Trainig Score
Train_score=model.score(x_train,y_train)
Train_score


# In[35]:


#Applying Cost-Complexity Pruning Path to get ccp_alpha values
path = model.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(clfs[-1].tree_.node_count, ccp_alphas[-1]))


# In[36]:


#Plotting visualization to help in  understanding  how the accuracy of the decision tree model
#changes as we vary the cost-complexity parameter (ccp_alpha)

train_scores = [clf.score(x_train ,y_train) for clf in clfs]
test_scores = [clf.score(x_test, y_test) for clf in clfs]

fig, ax = plt.subplots(figsize=(16,9))
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",drawstyle="steps-post")
ax.legend()
plt.show()


# In[37]:


ccp_alphas


# In[38]:


#Implementing  ccp_alpha value in Our model
clf1 = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=0.01959184)
clf1.fit(x_train,y_train)

pred_test1=clf1.predict(x_test)
pred_train1=clf1.predict(x_train)
print('Training Accuracy',accuracy_score(y_train, pred_train1),  'Testing Accuracy',accuracy_score(y_test, pred_test1))

#Plotting Confusion Matrix and Classifiction report to check accuracy
sns.heatmap(confusion_matrix(y_test, pred_test1),annot=True,fmt = "d",linecolor="k",linewidths=3)
print(classification_report(y_test,pred_test1))


# In[39]:


#Implementing  ccp_alpha value in Our model
clf2 = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=0.02429388)
clf2.fit(x_train,y_train)

pred_test2=clf2.predict(x_test)
pred_train2=clf2.predict(x_train)
print('Training Accuracy',accuracy_score(y_train, pred_train2),  'Testing Accuracy',accuracy_score(y_test, pred_test1))

#Plotting Confusion Matrix and Classifiction report to check accuracy
sns.heatmap(confusion_matrix(y_test, pred_test2),annot=True,fmt = "d",linecolor="k",linewidths=3)
print(classification_report(y_test,pred_test2))


# In[ ]:





# In[ ]:





# In[40]:


#Implementing  ccp_alpha value in Our model
clf = DecisionTreeClassifier(criterion='entropy',random_state=0, ccp_alpha=0.451395918367347)
clf.fit(x_train,y_train)


# In[41]:


pred_test=clf.predict(x_test)
pred_train=clf.predict(x_train)
print('Training Accuracy',accuracy_score(y_train, pred_train),  'Testing Accuracy',accuracy_score(y_test, pred_test))


# In[42]:


#Plotting Confusion Matrix and Classifiction report to check accuracy
sns.heatmap(confusion_matrix(y_test, pred_test),annot=True,fmt = "d",linecolor="k",linewidths=3)
print(classification_report(y_test,pred_test))


# In[43]:


#Plotting the Decision tree
plt.figure(figsize=(15,10))
tree.plot_tree(clf,filled=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




