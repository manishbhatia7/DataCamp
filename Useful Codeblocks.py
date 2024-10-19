# Check for Multicollinearity
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