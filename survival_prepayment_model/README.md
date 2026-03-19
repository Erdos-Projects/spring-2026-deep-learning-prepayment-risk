We train Random Forests, LSTM, and TabNet on FED data + non-borrower based loan features to predict prepayments and defaults using survival analysis.

Our two Random Forests were the best non-deep learning model observed. We chose to fit TabNet and LSTM as all-in-one models to predict prepayments and defaults simultaeously. TabNet acts as a "deep learning" random forest and is better with sparse datasets. LSTM generally excels at temporal/sequential data like our month-to-month loan data, though it prefers larger data sets. 

All models struggle to predict defaults. In another file, [we train DeepHit](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/tree/main/DeepHit) with prepayments censored to try and enhance default prediction. Analysis on DeepHit make it unlikely that censoring defaults will improve prepayment preformance in LSTM and TabNet, though this was not thoroughly explored. 

# **Results:**
**TabNet Model Results -  FED Data only (8 features + RF as feature) - All-in-one model**

Defaults: PR-AUC .015, 10th decile lift 1.2x

Prepayments: PR-AUC .006, 10th decile lift 1.3x

Notes: Performance improved on a smaller training set, but peaked by Epoch 4. Likely not rich enough data 


**LSTM Model Results - FED Data only (5 features)-All-in-one model**

Defaults: PR-AUC .011, 10th decile lift 1.4x, KS-Score 0.088

Prepayments: PR-AUC .003, 10th decile lift 2.8x - 3.1x (20 Epochs vs 30 Epochs), KS-Score .515


**RF Model Results - FED Data only (5 features) - 2 separate models**     

Defaults: PR-AUC 0.015, 10th decile lift 1.2x, KS-Score .051-.081 

Prepayments: PR-AUC .013, 10th decile lift 2.5x, KS-Score .51-.67  (.51 on reduced train set, .67 on larger train set



We conclude that overall performance and simplicity was dominated by our [Random Forest](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/Sample_Simulation/FinalRandomForest.ipynb). Random Forest dominance is likely due to the limited dataset, short data timeframe, extremely imbalanced data, and limited feature set. 
