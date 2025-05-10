# ML_project_dsaa2011
## Project Overview
This repository contains the code and documentation for the DSAA2011 course project at the Hong Kong University of Science and Technology (Guangzhou). The project focuses on analyzing the Dry Bean Dataset using various machine learning techniques.
## Dataset
Dataset Name: Dry Bean Dataset
Description: Contains 13,611 instances of dry bean measurements with 16 features (e.g., area, perimeter, major axis length, eccentricity) and 7 classes representing different bean varieties (e.g., Seker, Barbunya, Bombay).
## Project Tasks
### Data Preprocessing: Handle missing values, encode non-numeric data, and standardize features.
### Data Visualization: Use t-SNE to visualize high-dimensional data in 2D/3D space.
### Clustering Analysis: Apply clustering algorithms (e.g., K-means, hierarchical clustering) and evaluate the results.
### Prediction: Train and test supervised learning models (e.g., decision tree, logistic regression).
### Model Evaluation: Calculate metrics (accuracy, precision, recall, F1-score) and draw ROC curves.
### Open-ended Exploration: Experiment with additional machine learning techniques to improve models.

```text
Project
├── 1. Data Preprocessing
│   ├── StandardScaler, Labelencoder
│   ├── Augment (Gaussian noise and duplication)
│   └── remove outliers
│
├── 2. Visualization
│   ├── t- SNE (interactive)
│   ├── PCA
│   └── isomap
│   
├── 3. Cluster Analysis （include further exploration）
│
├── 4. Prediction （include further exploration）
│
└── 5. Evaluation and Choice of Prediction Model
            ├── ROC
            └── to be continued

```


```text
聚类算法（Clustering Methods）
├── 1. 基于原型的聚类（Prototype-based Clustering）
│   ├── K-Means
│   └── K-Medoids
│   
│
├── 2. 基于层次的聚类（Hierarchical Clustering）
│   ├── Agglomerative
│   └── Divisive
│
├── 3. 基于密度的聚类（Density-based Clustering）
│   ├── DBSCAN（核心点、边界点、噪声）
│   ├── OPTICS（解决DBSCAN对半径敏感的问题）
|   |      └── exploration (grid search / random search for fine-tuning hyperparameter) 
│   └── HDBSCAN（Hierarchical DBSCAN）
│
├── 4. 基于模型的聚类（Model-based Clustering）
│   ├── Gaussian Mixture Models（GMM，高斯混合模型）
│   └── Hidden Markov Models（HMM，用于序列聚类）
│
└── 5. Further exploration(combining method)
    ├── DEC (Deep Embedded Clustering)
    |    └──lower dimension embedding + kmeans cluster
    └── DeepCluster（深度表示 + 聚类）
         └── neural network extracting features + k-means cluster
```



```text
分类预测模型
├── 1. 线性模型（Linear Models）
│   └── Logistic Regression
|         ├── process the training data (outlier/ noise / duplication)
│         └── Ridge Logistic Regression (further exploration)
│
├── 2. 基于树的模型（Tree-based Models）
│   ├── Decision Tree
|   |      └── explore best hyperparameter
│   └──  Extra Tree
│   
├── 3. 支持向量机（SVM）
│   ├── Linear SVM
│   └── Kernel SVM（RBF、Polynomial）
│
├── 4. K近邻模型（K-Nearest Neighbors, KNN）
│
├── 5. 神经网络（Neural Networks） ......
│   ├── 全连接网络（MLP）？guess
│   ├── 卷积神经网络（CNN）
│   ├── 循环神经网络（RNN）
│   └──  Transformer
│
└── 6. 集成学习（Ensemble Learning）
    ├── 7.1 Bagging（Bootstrap Aggregating）
    │   ├── Random Forest
    │   ├── Bagging Classifier with SVC / KNN / Trees
    │   └── Extra Trees（Extremely Randomized Trees）
    │
    ├── 7.2 Boosting ......guess
    │   ├── AdaBoost
    │   ├── Gradient Boosting
    │   ├── XGBoost
    │   ├── LightGBM
    │   └── CatBoost
    │
    └── 7.4 Voting
        ├── Hard Voting（多数投票）
        └── Soft Voting（概率平均）
```








