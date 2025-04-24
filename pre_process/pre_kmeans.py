from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
#__file__ 是Python内置变量，表示当前脚本的完整路径


if current_dir not in sys.path:
    sys.path.append(current_dir)


datapath = os.path.join(current_dir, "Dry_Bean_Dataset.xlsx")
data = pd.read_excel(datapath, header=0)

# 特征和标签
X = data.drop(columns=['Class'])  # 只取特征，不要标签

print("start standardization")
# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("end standardization")

# # PCA降维（保留95%方差）
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# KMeans聚类（比如先设7类）
kmeans = KMeans(n_clusters=7, random_state=42)
#clusters = kmeans.fit_predict(X_pca)
clusters = kmeans.fit_predict(X_scaled)
print("聚类完成，预测标签长度:", len(clusters))

# 加回去看聚类结果
data['Cluster'] = clusters
print(data[['Cluster', 'Class']].head())
