import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))


if current_dir not in sys.path:
    sys.path.append(current_dir)


datapath = os.path.join(current_dir, "Dry_Bean_Dataset.xlsx")
data = pd.read_excel(datapath, header=0)

# print(data.head())

X = data.drop("Class", axis=1)
y = data["Class"]

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def remove_outliers_iqr(X, threshold=1.5):
    """通过IQR方法去除异常值，返回去除后的数据和保留的索引"""
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
    return X[mask], mask

def add_gaussian_noise(X, mean=0.0, std=0.01):
    """添加高斯噪声"""
    noise = np.random.normal(mean, std, X.shape)
    return X + noise

def augment_data(X, y, copies=1, noise_std=0.01):
    """数据增强：对每个样本加噪声复制若干次"""
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(copies):
        noisy_X = add_gaussian_noise(X, std=noise_std)
        augmented_X.append(noisy_X)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.vstack(augmented_y)

# ========== 新增步骤 ==========

# 去除异常值（基于IQR）
X_no_outliers, mask = remove_outliers_iqr(X_scaled, threshold=1.5)
y_encoded_no_outliers = y_encoded[mask]
print("去除异常值后数据形状:", X_no_outliers.shape)

X_train_no_outliers, X_test_no_outliers, y_train_no_outliers, y_test_no_outliers = train_test_split(
    X_no_outliers, y_encoded_no_outliers, test_size=0.2, random_state=42
)

# 添加噪声版本
X_noisy = add_gaussian_noise(X_no_outliers, std=0.02)

X_train_noisy, X_test_noisy, y_train_noisy, y_test_noisy = train_test_split(
    X_noisy, y_encoded_no_outliers, test_size=0.2, random_state=42
)


X_aug, y_aug = augment_data(X_no_outliers, y_encoded_no_outliers, copies=1, noise_std=0.015)

X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(                                                                  
        X_aug, y_aug, test_size=0.2, random_state=42
    )  



# 保存处理结果
np.save("src/pro_data/X_train_no_outliers.npy", X_train_no_outliers)
np.save("src/pro_data/X_test_no_outliers.npy", X_test_no_outliers)
np.save("src/pro_data/y_train_no_outliers.npy", y_train_no_outliers)
np.save("src/pro_data/y_test_no_outliers.npy", y_test_no_outliers)

np.save("src/pro_data/X_train_augmented.npy", X_train_aug)
np.save("src/pro_data/X_test_augmented.npy", X_test_aug)
np.save("src/pro_data/y_train_augmented.npy", y_train_aug)
np.save("src/pro_data/y_test_augmented.npy", y_test_aug)

np.save("src/pro_data/X_train_noisy.npy", X_train_noisy)
np.save("src/pro_data/X_test_noisy.npy", X_test_noisy)
np.save("src/pro_data/y_train_noisy.npy", y_train_noisy)
np.save("src/pro_data/y_test_noisy.npy", y_test_noisy)

# 如需可视化验证效果可用 matplotlib
# import matplotlib.pyplot as plt
# plt.scatter(X_pca_aug[:, 0], X_pca_aug[:, 1], c=np.argmax(y_aug[:100], axis=1))
# plt.show()

