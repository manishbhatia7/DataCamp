# Check for Multicollinearity
from sklearn.model_selection import GridSearchCV


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)

    return col_corr

# Plotting the ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

y_pred_probs=logreg.predict_proba(X_test)[:,1]
fpr,tpr,thresholds=roc_curve(y_test,y_pred_probs)
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr,label='ROC')
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC Curve')
show()

#ROC AUC
roc_auc_score(y_test,y_pred_probs)


#Grid Search CV
kf=KFold(n_splits=5,random_state=10,shuffle=True)
param_grid={"alpha":np.arange(0.0001,1,10),"solver":["sag","lsqr"]}
ridge=Ridge()
ridge_cv=GridSearchCV(ridge,param_grid,cv=kf)


#Create the parameter space
params = {"penalty": ["l1", "l2"],
         "tol": np.linspace(0.0001, 1.0, 50),
         "C": np.linspace(0.1,1.0,50),
         "class_weight": ["balanced", {0:0.8, 1:0.2}]}


#Dropping missing data with 5% of values
df.dropna(subset=['SalePrice'])

#Imputation
from sklearn.impute import SimpleImputerX_cat = music_df["genre"].values.reshape(-1, 1)
X_num = music_df.drop(["genre", "popularity"], axis=1).values
y = music_df["popularity"].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2,random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2,random_state=12)
imp_cat = SimpleImputer(strategy="most_frequent")
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)


