## General Basics

1. **What types are there for machine learning?**

   1. supervised
   2. unsupervised
   3. reinforcement - Reinforcement Learning is less supervised which depends on the agent in determining the output.

   </br>

## Preprocessing

1. **What is data preprocessing ?**

2. **What is Feature Scaling/Data Normalization?**

   rescale the data/features to put them into range [0,1]

3. What are different types of feature scaling?**

​		Standarization, MinMaxScaler, RobustScaler, Normalizer

4. **What is Standarization?**

5. **What models need standarization/rescaling?**

6. **What is data leakage and How do we correct it?**

7. **Ways to Encode Categorical Features. Cons and Pros**

8. **What is Discretization**

9. **Should we always remove missing values?**

10. **Why can't we just simply drop missing values?**

11. **What is KNN imputer**

12. **How do you handle outliers in your dataset**

13. **What is Isolation Forests and how it works?**

    similar as random forest, built based on decision trees. And It is built on the fact the outliers are few and different. We identify the outliers by choosing the samples ending up in shorter branches which indicate it was easier for the tree to separate them from other observations. 

</br>

## Linear Models

1. **What is Linear Regression? Pros and Cons**

2. **Why do we need Ridge(L2) Regression and what is it?**

   deal with multicolinearity

3. **What is L2 Regression assumption**

4. **What is L1 Regression, what's the difference with L2**

5. **If we want to reduce number of features , which one to choose, L1,L2?**

   Lasso

6. **What is ElasticNet, and how is it related to Lasso and Ridge?**

7. **What are the assumptions for linear regression?**

   - Linear relationship between independent and dependent variable
   - error terms constance varaince
   - error term follows normal distribution
   - no correlation among error terms

8. **What is VIF , and what is its cutoff value?**

   Check Multilinearity, 10

9. **Model evaluation metrics for Linear Regression**

   - MSE/MAE, MAE more robust to outliers
   - R^2 /Adjusted R2

10. **What is Logistic Regression, pro and cons?**

11. **Why not use MSE in Logistic?**

12. **Why log odds can predict probability?**

    because it used sigmoid function, basically squashed linear function $w^Tx$ between [0,1]

13. **Difference and similarity between logistic and linear regression**

14. **Difference between SVM and Logistic Regression**

    SVM tries to find the best margin that separates the classes and this reduce the risk of error on data, while logistic regression have different decision boundaries on different weights

    SVM works well on unstructured data such as text, images, but LR works better on identified independent variables.

    The risk of over fitting is less on SVM 

15. **What is KNN and how it's working?**

16. **What's the difference between K-means and KNN?**

    KNN represents a supervised classification algorithm that will give new data points accordingly to the k number or the closest data points, while k-means clustering is an unsupervised clustering algorithm that gathers and groups data into k number of clusters.

17. **How to choose K value in KNN**

    small k: complicate model, overfitting

    Large k: simple model, underfitting

</br>

## Trees

1. **How the tree will be split in decision trees ?**
2. **Why trees don't need standarization**

because scale of data won't affect how the trees be splited, in another word, it won't affect the ranking

3. **What's the problem with DT?**

4. **What is ensemble Learning and what types are there?**

5. **What is bootstrap, and how it works**

6. **How can bootstrap improve model**

7. **What is Random Forest and how it works? Pro and Cons**

8. **How to tune Random Forest?**

   1. Max_features
   2. Pre_prune

9. **What is the diffference and similarity between Out-of-Bag score and CV score**

10. **What is Boosting, and what are the common algorithms using boosting**

11. **What is Gradient Boosting , and how it works**

12. **Compare Gradient Boosting with Adaboosting**

13. **What are the advantages and disadvantages of GBDT**

14. **How to tune GBDT**

15. **What are the differences and similarities between Random Forest and GBDT**

16. **What is XGBoost**

17. **When do you want to use Trees instead of Linear Models**

18. **How trees are pruned?**

    https://medium.com/analytics-vidhya/post-pruning-and-pre-pruning-in-decision-tree-561f3df73e65

18. **Feature Importance**
19. 

</br>

## Model Evaluation

1. **What are the common metrics for Binary Classification**
2. **What does Precision and recall mean? why do we need F1 score?**
3. **What is Precision-Recall curve, which point is the best**
4. **What is ROC, AUC, which point is the best**
5. **What are the common metrics for multi-class classification**
6. **What are the common metrics for regression**
7. **What is imbalanced data, and what  are the two sources of them**
8. **Two basic approaches in dealing with imbalanced data**
9. **what is SMOTE**
10. **How cross-validation works**
11. **What is bias , what does it mean to have high bias**
12. **What is variance ,  what does it mean to have high variance**
13. **What is bayesian error**
14. **what is the total error of a model**
15. **How to trade off variance and bias**
16. 

</br>

## Clustering

1. **What is Kmeans, and how it works**

   We want to add k new points to the data we have, each one of those points is call centroid, will be going around and trying to find a center in the middle of the k clusters we have. Algorithm stops when the points stop moving

   Hyperparameter: k – number of clusters we want

   Steps:

   Assign initial values to each cluster 

   Repeat: assign each point from training to each cluster that is closest to it in mean, recalculate the mean of each cluster

   If all clusters’ mean stop changing, algorithm stops

2. **What are K-means' Pros and Cons**

   **pro**: 

   - easy to implement and tune with hyper parameter k. 
   - it guarantees convergence and easily adapt to new examples
   - low computational cost

   **cons**: 

   - Centroids can be dragged by outliers, or outliers might get their own cluster instead of being ignored. 
   - K-means algorithm can be performed in numerical data only.

3. **How to choose K, what metric**

   minimize variance within cluster, maximize variance with clusters

4. **What kind of clusters do you know**
5. **What is distortion function? Is it convex or non-convex?**
6. **Tell me about the convergence of the distortion function.**
7. **Topic: EM algorithm**
8. **What is the Gaussian Mixture Model?**
9. **Describe the EM algorithm intuitively.**
10. **What are the two steps of the EM algorithm**
11. **Compare GMM vs GDA.**

</br>

## Dimensionality Reduction

1. **Why do we need dimensionality reduction techniques?** 

   data compression, speeds up learning algorithm and visualizing data

2. **What do we need PCA and what does it do? **

   PCA tries to find a lower dimensional surface project , such that the sum of the squared projection error is minimized

3. **What is the difference between logistic regression and PCA?**

4. **What are the two pre-processing steps that should be applied before doing PCA?**

    mean normalization and feature scaling

5. **If we want only the 90% varaince, how to choose PC**
6. **PCA calculation details**
7. 







