This [sample simulation](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/Sample_Simulation/LoanPoolSim.ipynb) uses models developed in this project to create a Monte-Carlo style simulation of cash flows. 
Using the [SARIMAX model](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/tree/main/Interest_Mean_Prediction), we generate average interest rates for loans beginning in 2022.
Using these interest rates, we create a pool of 500 small personal loans
We then run a cash flow simulation beginning 2022 by choosing whether a loan prepays/defaults/continues each month based on our [Random Forest's](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/Sample_Simulation/FinalRandomForest.ipynb) prediction and using a Monte-Carlo style decision mechanism. 
The result is the Random Forest overestimates the defaults in late 2022-early 2023, which is an exaggeration of the actual upticks in defaults. This could be from RF's poor fit on defaults, or the inclusion of 2008 in the training data. 

Future work includes integrating [our DeepHit model for defaults](https://github.com/Erdos-Projects/spring-2026-deep-learning-prepayment-risk/blob/main/DeepHit/DeepHit_SingleRisk.ipynb) (in place of the RF) and expanding training data to a wider economic scape. 
