import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
import seaborn as sns
from collections import Counter

# 1. get the training file
file_path = 'Dry_Bean_Dataset.xlsx'
data = pd.read_excel(file_path)

# 2. divide the features and labels
X = data.iloc[:, :-1].values  
y = data.iloc[:, -1].values

# 3. process the non-numeric label
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

label_counts = Counter(y)
print("=== 类别样本数量统计 ===")
for label, count in label_counts.items():
    print(f"label '{label}' (encode: {label_mapping[label]}): {count} samples")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# 4. Standardlization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

#分类器训练：分别训练随机森林和XGBoost分类器，并通过10折交叉验证来评估模型的性能。
#集成分类器：将随机森林和XGBoost的预测概率进行组合，通过选择最高概率对应的类别作为最终预测结果。
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_predict, cross_validate

# 1. 训练随机森林分类器并进行10折交叉验证
rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced'
)

# 10折交叉验证评估随机森林
rf_cv_results = cross_validate(
    rf_clf, 
    X_train, 
    y_train,
    cv=10,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True,
    return_estimator=True
)

print("Random Forest Cross-Validation Results:")
print(f"Mean Train Accuracy: {np.mean(rf_cv_results['train_accuracy']):.4f}")
print(f"Mean Test Accuracy: {np.mean(rf_cv_results['test_accuracy']):.4f}")
print(f"Mean Test Precision: {np.mean(rf_cv_results['test_precision_weighted']):.4f}")
print(f"Mean Test Recall: {np.mean(rf_cv_results['test_recall_weighted']):.4f}")
print(f"Mean Test F1-score: {np.mean(rf_cv_results['test_f1_weighted']):.4f}")

# 2. 训练XGBoost分类器并进行10折交叉验证
xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=7,
    learning_rate=0.1,
    max_depth=20,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42,
    eval_metric='mlogloss'
)

# 10折交叉验证评估XGBoost
xgb_cv_results = cross_validate(
    xgb_clf,
    X_train,
    y_train,
    cv=10,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True,
    return_estimator=True
)

print("\nXGBoost Cross-Validation Results:")
print(f"Mean Train Accuracy: {np.mean(xgb_cv_results['train_accuracy']):.4f}")
print(f"Mean Test Accuracy: {np.mean(xgb_cv_results['test_accuracy']):.4f}")
print(f"Mean Test Precision: {np.mean(xgb_cv_results['test_precision_weighted']):.4f}")
print(f"Mean Test Recall: {np.mean(xgb_cv_results['test_recall_weighted']):.4f}")
print(f"Mean Test F1-score: {np.mean(xgb_cv_results['test_f1_weighted']):.4f}")

# 3. 训练最终模型（在整个训练集上）
rf_clf.fit(X_train, y_train)
xgb_clf.fit(X_train, y_train)

from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[('rf', rf_clf), ('xgb', xgb_clf)],
    final_estimator=LogisticRegression(),  # 使用元模型组合预测
    cv=5
)
stacking_clf.fit(X_train_resampled, y_train_resampled)
y_pred_voting = stacking_clf.predict(X_test)

# 计算集成模型的评估指标
accuracy = accuracy_score(y_test, y_pred_voting)
precision = precision_score(y_test, y_pred_voting, average='weighted')
recall = recall_score(y_test, y_pred_voting, average='weighted')
f1 = f1_score(y_test, y_pred_voting, average='weighted')

print("\nEnsemble Model Performance on Test Set:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

#XGBoost
# 转换为XGBoost的DMatrix格式（优化数据加载）
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置模型参数（关键参数）
params = {
    'objective': 'multi:softmax',  # 多分类目标函数
    'num_class': 7,                # 类别数
    'learning_rate': 0.1,          # 学习率
    'max_depth': 20,               # 树的最大深度
    'subsample': 0.7,              # 样本采样比例
    'colsample_bytree': 0.8,       # 特征采样比例
    'seed': 42,                    # 随机种子
    'eval_metric': 'mlogloss'      # 多分类对数损失
}

# 训练模型
num_round = 100  # 迭代轮数
model = xgb.train(params, dtrain, num_round)

# 预测与评估
y_pred = model.predict(dtest)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# 特征重要性可视化
xgb.plot_importance(model)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Draw ROC and calculate AUC
# For multi-class, we need to compute the ROC curve and AUC for each class
# Here we assume a binary classification for simplicity
if len(np.unique(y_test)) == 2:
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(dtest)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
else:
    # For multi-class, compute ROC and AUC for each class
    from sklearn.preprocessing import label_binarize

    # Binarize the output
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    n_classes = y_test_binarized.shape[1]

    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(dtest, output_margin=True)
    y_proba = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'
                                              ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for Multi-class')
    plt.legend(loc="lower right")
    plt.show()


#CatBoost
from catboost import CatBoostClassifier, Pool
# 定义 CatBoost 模型
model = CatBoostClassifier(
    iterations=1000,           # 迭代次数
    learning_rate=0.1,         # 学习率
    depth=6,                  # 树深度
    loss_function='MultiClass', # 多分类损失函数
    eval_metric='Accuracy',    # 评估指标
    verbose=100,               # 每100轮打印一次日志
    random_state=42
)

# 训练模型
model.fit( X_train, y_train, eval_set=(X_test, y_test), plot=False, early_stopping_rounds=50)

# 模型评估
y_pred = model.predict(X_test)
print("\n分类报告：")
print(classification_report(y_test, y_pred))

#Adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier  # AdaBoost默认使用决策树作为基学习器


# 定义AdaBoost模型
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=12),  # 基学习器（默认是决策树，可自定义）
    n_estimators=200,                              # 弱学习器数量
    learning_rate=0.5,                             # 学习率（缩减每个弱学习器的贡献）
    algorithm='SAMME.R',                           # 多分类算法（SAMME.R更高效）
    random_state=42
)

# 训练模型
adaboost.fit(X_train, y_train)

# 模型评估
y_pred = adaboost.predict(X_test)
print("分类报告：\n", classification_report(y_test, y_pred))
