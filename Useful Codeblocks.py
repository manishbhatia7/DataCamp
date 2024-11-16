# Check for Multicollinearity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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

##Evaluate Multiple Models
#Training Model
models={"Logistic Regression":LogisticRegression(),"KNN":KNeighborsClassifier(),"Decision Tree Classifier":DecisionTreeClassifier()}
results=[]
for model in models.values():
    kf=KFold(n_splits=5,random_state=10,shuffle=True)
    cv_results=cross_val_score(model,X_train,y_train,cv=kf)
    results.append(cv_results)
    plt.boxplot(results,labels=models.keys())


#Testing Model Performance on Test Sets
for name,model in models.items():
    model.fit(X_train,y_train)
    test_score=model.score(X_test,y_test)
    print(name,test_score)

#Preprocessing Model
# Create steps
steps = [("imp_mean",SimpleImputer()),
         ("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga", "lbfgs"],
         "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(tuning.best_params_,tuning.score(X_test,y_test)))

#TSNE
from matplotlib.pyplot import plt
from sklearn.manifold import TSNE
model=TSNE(learning_rate=100)
transformed=model.fit_transform(samples)
xs=transformed[:,0]
ys=transformed[:,1]
plt.scatter(xs,ys)

#Transforming Categorical Variables using pandas and prefixing with a specific variable
df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)

#Detecting the optimal no of clusters using Kmeans
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_preprocessed)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 10), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#Calculate Floor
nobel['decade']=(np.floor(nobel['year'] / 10) * 10).astype(int)

#Defining threshold for missing values ideally it is 5%
threshold=len(col)*0.05
cols_to_drop=df.columns[df.isna().sum()<=threshold]
df.dropna(subset=cols_to_drop, inplace=True)

#Imputing a Summary Statistic
cols_with_missing_values=df.columns[df.isna().sum()>0]
for col in cols_with_missing_values:
    df[col].fillna(df[col].mean())


    #Feature Selection using Lasso
    # Specify L1 regularization
    lr = LogisticRegression(solver='liblinear', penalty='l1')

    # Instantiate the GridSearchCV object and run the search
    searcher = GridSearchCV(lr, {'C': [0.001, 0.01, 0.1, 1, 10]})
    searcher.fit(X_train, y_train)

    # Report the best parameters
    print("Best CV params", searcher.best_params_)

    # Find the number of nonzero coefficients (selected features)
    best_lr = searcher.best_estimator_
    coefs = best_lr.coef_
    print("Total number of features:", coefs.size)
    print("Number of selected features:", np.count_nonzero(coefs))
