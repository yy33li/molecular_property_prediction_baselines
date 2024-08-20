# import modules
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, auc, precision_recall_curve
import math
import numpy as np
from scipy.stats import pearsonr

#read prediction output files
test_predict = pd.read_csv('prediction_path')

#read test file to get label  
test_observ = pd.read_csv('test_path')

#merge to two files two ensure prediction and label are match based on SMILES
merged_df = pd.merge(test_observ, test_label, on='smiles')

l1 = merged_df['label']
l2 = merged_df['predictions']

#for classifcation tasks
##create a for loop function to take the prediction average of all prediction tasks within a dataset as final score
roc_auc_all = []
prc_auc_all = []

for i in test_label.columns.tolist():
    if i != 'smiles':
        obs = test_observ[i]
        empty = []
        for idx, smi in enumerate(obs):
            if math.isnan(smi) == True:
                empty.append(test_observ['smiles'][idx])
        
        #remove all empty values       
        test_observ_new = test_observ[~test_observ['smiles'].isin(empty)]
        l1 = test_observ_new[i]
    
        test_predict_new = test_predict[~test_predict['smiles'].isin(empty)]
        l2 = test_label_new[i]
        
        #calculate ROC-AUC
        roc_auc = roc_auc_score(l1, l2)
        roc_auc_all.append(roc_auc)
        
        #calculate PR-AUC
        prec, recall, _ = precision_recall_curve(l1, l2)
        prc_auc = auc(recall,prec)    
        prc_auc_all.append(prc_auc)        
        


roc_auc = sum(roc_auc_all)/len(roc_auc_all)
pr_auc = sum(prc_auc_all)/len(prc_auc_all)


#for regression task

##calculate RMSE
mse = mean_squared_error(l1, l2)
rmse = math.sqrt(mse)

##claculate pearson correlation
corr, _ = pearsonr(l1, l2)
correlation = corr
    
        
