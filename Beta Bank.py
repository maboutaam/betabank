# # Beta Bank 

# ### Since Beta Bank would rather hang onto its current clientele than draw in new ones, we must forecast if a given client will quit in the near future for this project.

# In[2]:


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


# ### Libraries are imported.

# In[3]:


# Data Laoding

df = pd.read_csv('/datasets/Churn.csv')


# ### The above dataframe contains information about Beta Bank's current customers.

# In[4]:


# working code

df.info()


# ### There is a total of 14 columns in this dataframe.

# In[5]:


# working code

df.describe()


# In[6]:


# working code

df.dtypes


# In[7]:


# working code

df.isnull().sum()


# In[8]:


# Handle Missing Values in 'Tenure'

# working code

median_tenure = df['Tenure'].median()
df['Tenure'].fillna(median_tenure, inplace=True)


# ### By using the above code we addressed the problem of having missing values in 'Tenure' column.

# In[9]:


# Convert Categorical Features

# working code

df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)


# In[10]:


# working code

df.columns


# In[32]:


# Select the Features and Target Variable

features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male']
target = 'Exited'

# Check if all feature columns are present
for feature in features:
    if feature not in df.columns:
        print(f"Column {feature} not found in the DataFrame.")
        
X = df[features]
y = df[target]


# ### We used all the columns as the features and 'Exited' as our target for this study.

# In[12]:


# split the data

# working code

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Standardize the feartures

# working code

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# <div class="alert alert-block alert-success">
# <b>Reviewer's comment</b> <a class="tocSkip"></a>
# 
# Correct.
# </div>

# In[14]:


# Examine the Balance of Classes

# working code

class_distribution = y.value_counts()
print(class_distribution)

# Plot the class distribution
sns.countplot(x=y)
plt.title('Class Distribution of Target Variable')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()


# ### According to the above bar graph. 7963 customers did not exist the bank and 2037 customers exited the bank. 

# In[15]:


# Train a Logistic Regression Model

# working code

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# ### Because of this imbalance, the Logistic Regression model performs well for the majority class but poorly for the minority class. Techniques like class weighting, oversampling the minority class, or employing more complex models may be required to increase performance on the minority class.

# In[16]:


# Approach 1: Oversampling the Minority Class using SMOTE

# Apply SMOTE to the Training Set

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# ### We are using SMOTE, a machine learning technique, to weaken the imbalance problem.

# In[33]:


# Train a Logistic Regression Model on SMOTE Data and Evaluate the Model Trained on SMOTE Data

model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_smote.predict(X_test)

print(classification_report(y_test, y_pred_smote))

conf_matrix_smote = confusion_matrix(y_test, y_pred_smote)
sns.heatmap(conf_matrix_smote, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (SMOTE)')
plt.show()


# ### Class 1 (exited) has a respectable recall rate of 71%, indicating that a sizable percentage of consumers who genuinely left are captured by the model. The F1 score in class 1 is lower compared to class 0 which means there is room for improvement. 393 instances were used. 75% of all predictions were correct.

# In[34]:


# Approach 2: Class Weighting in Logistic Regression

# Train a Logistic Regression Model with Class Weighting and Evaluate the Model with Class Weighting

model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)

y_pred_weighted = model_weighted.predict(X_test)

print(classification_report(y_test, y_pred_weighted))

conf_matrix_weighted = confusion_matrix(y_test, y_pred_weighted)
sns.heatmap(conf_matrix_weighted, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Class Weighting)')
plt.show()


# ### Overall, both methods are quite similar in performance, but SMOTE has a slight edge in terms of recall and F1-score, making it the slightly better approach for this specific dataset and task. 393 instances were used. 78% of predictions were correct.

# In[19]:


# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


# In[20]:


# Standardize features

scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)


# In[24]:


# Final Testing

param_grid = {
       'n_estimators': [100, 200],
       'max_depth': [None, 10],
       'min_samples_split': [2, 5],
       'min_samples_leaf': [1, 2],
       'class_weight': ['balanced', None]
   }
   
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid, 
    cv=3,
    scoring='f1', 
    verbose=2, 
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train_smote, y_train_smote)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# In[25]:


# working code

print(type(X_train_smote), X_train_smote.shape)
print(type(y_train_smote), y_train_smote.shape)


# In[26]:


# working code

print(type(X_train_smote), X_train_smote.shape)
print(type(y_train_smote), y_train_smote.shape)
print(type(X_test), X_test.shape)
print(type(y_test), y_test.shape)


# In[31]:


# Final Testing

best_rf_model = grid_search.best_estimator_
y_pred_test = best_rf_model.predict(X_test)
   
# Evaluate Performance
print(classification_report(y_test, y_pred_test))
   
f1_test = f1_score(y_test, y_pred_test)
print("F1 Score for Test Set: {:.4f}".format(f1_test))
   
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Test Set)')
plt.show()


# ### F1 score is 0.648. For Precision is Class 1, 61% of predicted positives were correct. While in Recall in Class 1, 61% of acutal positives were correctly indified. There was a total of 393 instances. 84% of all instances were correct.

# In[30]:


# Measurement the AUC-ROC metric and comparison it with the F1.

# Compute predicted probabilities for class 1 (Exited) for AUC-ROC
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score for Test Set: {auc_roc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.4f})'.format(auc_roc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print F1 score for comparison
f1_test = f1_score(y_test, y_pred_test)
print(f"F1 Score for Test Set: {f1_test:.4f}")


# ### As seen in the above curve, AUC is high because ROC is close to the upper left corner. The AUC value is approximately 1. This means that the model is performing well.


# # Conclusion
# ### According to the Class Distribution of Target Variable, 7963 clients did not exist the bank and 2037 clients exited the bank.
# ### The imbalance is very common among organizations and this is normal process of business. However, this problem has to be address.
# ### Regression model performs well for the majority class but poorly for the minority class. 
# ### We used two approaches: SMOTE and Class Weighting in Logistic Regression.
# ### In SMOTE F1 it showed that there is room for improvement.
# ### In Class Weighting in Logistic Regression, we had a recall rate of 61%.
# ### Both approaches are similar in results but SMOTE have better recall rate.
# ### According to the Confusion Matrix for the Final Testing, F1 = 0.6048 which is higher than 0.59.
# ### The AUC-ROC score is 0.8530.
# ### There is a real need to reduce the imbalance in order to keep more clients which could lead to attraction of new clients and therefore growth for the bank.
# ### Age which is one the features could have a significant impact on the imbalance results which means the bank should make market campaigns that directly targets younger people because they tend to stay more on the long term.
# ### Also Balance and CreditScore gives and insight on the type of clients the bank is looking to have. 
# ### The Bank has to make a market differention and assume who are their target clients in the market.
# ### A score of 0.8530 for AUC-ROC is relatively goof, because this means that the model have a strong ability to know which clients would churn and who would not.
# ### F1 = 0.6048 suggests that the model is performing well at balancing precision and recall, but there is room for improvement.


# In[ ]:




