# Load requires libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(r"C:\Users\dhana\Desktop\lending_club_loan_two.csv")
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 396030 entries, 0 to 396029
Data columns (total 27 columns):
 #   Column                Non-Null Count   Dtype  
---  ------                --------------   -----  
 0   loan_amnt             396030 non-null  float64
 1   term                  396030 non-null  object 
 2   int_rate              396030 non-null  float64
 3   installment           396030 non-null  float64
 4   grade                 396030 non-null  object 
 5   sub_grade             396030 non-null  object 
 6   emp_title             373103 non-null  object 
 7   emp_length            377729 non-null  object 
 8   home_ownership        396030 non-null  object 
 9   annual_inc            396030 non-null  float64
 10  verification_status   396030 non-null  object 
 11  issue_d               396030 non-null  object 
 12  loan_status           396030 non-null  object 
 13  purpose               396030 non-null  object 
 14  title                 394274 non-null  object 
 15  dti                   396030 non-null  float64
 16  earliest_cr_line      396030 non-null  object 
 17  open_acc              396030 non-null  float64
 18  pub_rec               396030 non-null  float64
 19  revol_bal             396030 non-null  float64
 20  revol_util            395754 non-null  float64
 21  total_acc             396030 non-null  float64
 22  initial_list_status   396030 non-null  object 
 23  application_type      396030 non-null  object 
 24  mort_acc              358235 non-null  float64
 25  pub_rec_bankruptcies  395495 non-null  float64
 26  address               396030 non-null  object 
dtypes: float64(12), object(15)
memory usage: 81.6+ MB
df.head(5)


df.describe().T

df.isnull().sum()
loan_amnt                   0
term                        0
int_rate                    0
installment                 0
grade                       0
sub_grade                   0
emp_title               22927
emp_length              18301
home_ownership              0
annual_inc                  0
verification_status         0
issue_d                     0
loan_status                 0
purpose                     0
title                    1756
dti                         0
earliest_cr_line            0
open_acc                    0
pub_rec                     0
revol_bal                   0
revol_util                276
total_acc                   0
initial_list_status         0
application_type            0
mort_acc                37795
pub_rec_bankruptcies      535
address                     0
dtype: int64
df.dropna(axis=0,inplace=True)
df.shape
(335867, 27)
#Correlations
df_numerics = df.select_dtypes(include=np.number)
df_numerics.corr()

#Age category
fig, ax = plt.subplots(figsize=(4, 4))
df["loan_status"].value_counts().head(10).sort_values(ascending=True).plot(
    kind="barh", color='r', ax=ax, title="loan_status")
ax.set_xlabel("loan_status")
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(x='loan_status', y = 'installment', data = df, errwidth=1,saturation=1)
plt.title('Loan Status vs installment \n')
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(x='loan_status', y = 'loan_amnt', data = df, errwidth=1,saturation=1)
plt.title('Loan Status vs loan amount \n')
plt.show()

sns.countplot(x= 'application_type', hue=df['loan_status'], data=df)

sns.countplot(x= 'purpose', hue=df['loan_status'], data=df)

X = df.drop(['loan_status'], axis=1)
y = df['loan_status']
from sklearn.preprocessing import StandardScaler,OneHotEncoder, LabelEncoder
encoder = LabelEncoder()
scaler = StandardScaler()
# Encoding the categorical features
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    X[col] = encoder.fit_transform(X[col])
X[categorical_features] = scaler.fit_transform(X[categorical_features])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12, shuffle=True)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
log = LogisticRegression(solver='liblinear')
log.fit(X_train,y_train)

LogisticRegression
LogisticRegression(solver='liblinear')
y_pred_log = log.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
print(f"Accuracy Score of Logistic Regression is : {acc_log}")

from sklearn import metrics
confusion_matrix_log = metrics.confusion_matrix(y_test, y_pred_log)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix_log, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=9, max_features="log2", n_estimators=25)
rf.fit(X_train, y_train)

RandomForestClassifier
RandomForestClassifier(max_depth=9, max_features='log2', n_estimators=25)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy score of Random forest is:", accuracy_rf)

confusion_matrix_rf = metrics.confusion_matrix(y_test, y_pred_rf)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix_rf, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()
No description has been provided for this image
KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

KNeighborsClassifier
KNeighborsClassifier()
y_pred_knn = knn.predict(X_test)
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy Score of KNN Classifier is : {acc_knn}")
Accuracy Score of KNN Classifier is : 0.7695834697948611
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test, y_pred_knn)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()
No description has been provided for this image
#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

GradientBoostingClassifier
GradientBoostingClassifier()
gb_pred = gb.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"Accuracy Score of Gradient Boosting Classifier is : {gb_acc}")
Accuracy Score of Gradient Boosting Classifier is : 0.8068746836573675
confusion_matrix_gb = metrics.confusion_matrix(y_test, gb_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix_gb, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()
No description has been provided for this image
# import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-white')
plt.figure(figsize=(15, 5))
models = ['Random Forest','Logistic Regression','Gradient Boosting','KNN']
test_accuracy = [accuracy_rf, acc_log, gb_acc, acc_knn,]
plt.plot(models, test_accuracy, color='red')
plt.ylim(0.70, 0.90)
plt.ylabel("Accuracy Score")
plt.show()
No description has been provided for this image
Model Hyperparameter Tune using Grid Search

Random Forest Model

param_grid = {'n_estimators': [25],
              'max_features': ['sqrt'],
              'max_depth': [3],
              'max_leaf_nodes': [3],
}
from sklearn.model_selection import GridSearchCV
grid_search_rf = GridSearchCV(RandomForestClassifier(),
                           param_grid=param_grid)
grid_search_rf.fit(X_train, y_train)
print(grid_search_rf.best_estimator_)
RandomForestClassifier(max_depth=3, max_leaf_nodes=3, n_estimators=25)
rf_hyper_pred = grid_search_rf.predict(X_test)
# Accuracy Score on test dataset
rf_accuracy_test = accuracy_score(y_test,rf_hyper_pred)
print('accuracy_score on test dataset : ', rf_accuracy_test)
accuracy_score on test dataset :  0.8035400601423169
y_rf_prob = grid_search_rf.predict_proba(X_test)[:, 1] 
from sklearn.metrics import roc_curve, roc_auc_score 
from sklearn.metrics import auc
fpr, tpr, thresholds = roc_curve(y_test, y_rf_prob, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_auc_rf = roc_auc_score(y_test, y_rf_prob) 
roc_auc_rf
0.6972865607602656
# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_rf) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [1, 0], 'k--', label='Random Forest Classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

confusion_matrix = metrics.confusion_matrix(y_test, rf_hyper_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()

Logistic Regression

param_grid = [    
    {'penalty' : ['l1'],
    'solver' : ['liblinear'],
    'max_iter' : [100]
    }
]
Log_tune = GridSearchCV(log, param_grid = param_grid)
Log_tune.fit(X_train, y_train)


LogisticRegression
log_tune_y_pred = Log_tune.predict(X_test)
# Accuracy Score on test dataset
log_tune_acc = accuracy_score(y_test,log_tune_y_pred)
print('accuracy_score on test dataset : ', log_tune_acc)

# compute ROC AUC 
y_log_prob = Log_tune.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_log_prob, pos_label=1)
roc_auc_log = roc_auc_score(y_test, y_log_prob) 

# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_log) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Logistic Regression classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()

log_confusion_matrix = metrics.confusion_matrix(y_test, log_tune_y_pred)
cm_display = metrics.ConfusionMatrixDisplay(log_confusion_matrix, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()

Gradient Boosting tune

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50],
    'learning_rate': [0.01],
    'max_depth': [3],
}
# Initialize GridSearchCV
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid, scoring='accuracy')
# Fit the model to the training data using GridSearchCV
grid_search_gb.fit(X_train, y_train)


GradientBoostingClassifier
# Make predictions on the test set using the best model
y_pred_gb_tune = grid_search_gb.predict(X_test)
# Accuracy Score on test dataset
gb_tune_acc = accuracy_score(y_test,y_pred_gb_tune)
print('accuracy_score on test dataset : ', gb_tune_acc)

gb_confusion_matrix = metrics.confusion_matrix(y_test, y_pred_gb_tune)
cm_display = metrics.ConfusionMatrixDisplay(gb_confusion_matrix, display_labels = ['Charged Off', 'Fully Paid'])
cm_display.plot()
plt.show()

#compute ROC AUC 
y_gb_prob = grid_search_gb.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_gb_prob, pos_label=1)
roc_auc_gb = roc_auc_score(y_test, y_gb_prob) 
roc_auc_gb

# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc_gb) 
# roc curve for tpr = fpr  
plt.plot([0, 1], [0, 1], 'k--', label='Gradient Boosting classifier') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('ROC Curve') 
plt.legend(loc="lower right") 
plt.show()


 
