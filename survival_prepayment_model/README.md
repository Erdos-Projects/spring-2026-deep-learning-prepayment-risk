We train Random Forests, LSTM, and TabNet on FED data + non-borrower based loan features to predict prepayments and defaults using survival analysis.

Our two Random Forests were the best non-deep learning model observed. We chose to fit TabNet and LSTM as an all-in-one model to predict prepayments and defaults simultaeously. TabNet acts as a "deep learning" random forest and is better with sparse datasets. LSTM generally excels at temporal/sequential data like our month-to-month loan data, though it prefers larger data sets. 

We conclude that overall performance and simplicity was dominated by our [Random Forest](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/Sample_Simulation/FinalRandomForest.ipynb). This is likely due to the limited dataset, short data timeframe, and limited feature set. 
