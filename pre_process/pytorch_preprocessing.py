import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


data = pd.read_excel("Dry_Bean_Dataset.xlsx")


X = data.drop("Class", axis=1)
y = data["Class"]

# 3. 标签编码为整数（用于 CrossEntropyLoss）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # y will be integer labels

# 4. 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. 分割训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# 6. 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 7. 打包成 Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

print("Classes:", label_encoder.classes_)
