Gussian distribution 
Density estimation 


Steps for anaomaly detection algorithm 

1. Choose n features xi that you think might be indicative of anomalous examples
2. Fit parameters u1, ... un, sigma2_1,.....,sigma2_n
3. Given new example, compute p(x)
4. Anomaly if p(x) < epsilon 

Evaluation 

1. Fit model p(x) on training set x1, x2...., xn
2. On a cross validation/test example x, predict
    y=0 
    y=1

============================================================================================================

Anomaly Detection vs Supervised learning

Anomaly Detection - very small number of positives examples and large number of negatives. eg: Fraud
Supervised Learning - Large number of positives and negative examples. eg: Spam