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

# cluster algorithm don't need this kind of preprocessing
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

print("start preprocessing")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("split data")
# X_train_cluster, X_test_cluster = train_test_split(
#     X_scaled, test_size=0.2, random_state=42
# )
# 同时分割，确保对应关系
X_train_cluster, X_test_cluster, y_encoded_train, y_encoded_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)




print(np.isnan(X_scaled).any())
print("是否有inf:", np.isinf(X_scaled).any())
# X_pca = PCA(n_components=0.95).fit_transform(X_scaled) 
#保留数据的主要信息（95%方差），维度数自适应
X_pca = PCA(n_components=2, svd_solver="randomized").fit_transform(X_scaled[:100])
#for visualization,确定是降到2维

print("X_train_cluster shape:", X_train_cluster.shape)


np.save("../pro_data/X_scaled_cluster.npy", X_scaled)
np.save("../pro_data/X_pca_cluster.npy", X_pca)
np.save("../pro_data/y_encoded.npy", y_encoded)

# np.save("X_train_cluster.npy", X_train_cluster)
# np.save("X_test_cluster.npy", X_test_cluster)
# np.save("y_encoded_train_cluster.npy", y_encoded_train)
# np.save("y_encoded_test_cluster.npy", y_encoded_test)
# print("successfully saved")

