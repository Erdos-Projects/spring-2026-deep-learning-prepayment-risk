We train Random Forests, LSTM, and TabNet on FED data + non-borrower based loan features to predict prepayments and defaults using survival analysis.

Our two Random Forests were the best non-deep learning model observed. We chose to fit TabNet and LSTM as all-in-one models to predict prepayments and defaults simultaeously. TabNet acts as a "deep learning" random forest and is better with sparse datasets. LSTM generally excels at temporal/sequential data like our month-to-month loan data, though it prefers larger data sets. 

All models struggle to predict defaults. In another file, [we train DeepHit](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/tree/main/DeepHit) with prepayments censored to try and enhance default prediction. Analysis on DeepHit make it unlikely that censoring defaults will improve prepayment preformance in LSTM and TabNet, though this was not thoroughly explored. 

We conclude that overall performance and simplicity was dominated by our [Random Forest](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/Sample_Simulation/FinalRandomForest.ipynb) with a PR-AUC of 013 and a KS-Score of .51-.67 (dependent on training set). Random Forest dominance is likely due to the limited dataset, short data timeframe, extremely imbalanced data, and limited feature set. 
