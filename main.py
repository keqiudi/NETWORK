import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# 过滤特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 加载数据
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 分离特征和标签
X_train = train_data.drop(columns=['id', 'attack_cat'])
y_train = train_data['attack_cat']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# 定义类别型特征列
categorical_cols = ['proto', 'service', 'state']

# 修改低频类别合并逻辑（原阈值50改为动态调整）
for col in categorical_cols:
    freq = X_train[col].value_counts()
    # 动态阈值：至少保留覆盖95%样本的类别
    cum_ratio = freq.cumsum() / freq.sum()
    threshold = freq[cum_ratio < 0.95].min()
    X_train[col] = X_train[col].apply(lambda x: x if freq[x] >= threshold else 'other')
    X_test[col] = X_test[col].apply(lambda x: x if x in freq and freq[x] >= threshold else 'other')

# 在OneHot编码前添加以下代码
# ============ 新增特征交互 ============
for df in [X_train, X_test]:
    # 创建协议与服务类型的组合特征
    df['proto_service'] = df['proto'] + '_' + df['service']

# 更新类别型特征列
categorical_cols = ['proto', 'service', 'state', 'proto_service']

# OneHot编码
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[categorical_cols])

# 转换数据集
def encode_features(df, encoder):
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

X_train_processed = encode_features(X_train, encoder)
X_test_processed = encode_features(X_test, encoder)

# 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# 划分验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train_encoded,
    test_size=0.2,
    stratify=y_train_encoded,
    random_state=42
)

# ============ 修改后的过采样部分 ============
from imblearn.over_sampling import SMOTE
from collections import Counter

# 获取训练集中各类别样本数
original_counts = Counter(y_tr)
print("Original Class Distribution:", original_counts)

# 计算弱势类目标样本数（至少为原始数量的1.5倍）
weak_classes = ['Analysis', 'Backdoor', 'DoS', 'Worms']
weak_indices = [i for i, name in enumerate(label_encoder.classes_) if name in weak_classes]

sampling_strategy = {
    i: max(int(1.5 * original_counts[i]), original_counts[i] + 1)
    for i in weak_indices
}
print("Adjusted Sampling Strategy:", sampling_strategy)

# 初始化SMOTE
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=5,  # 减少近邻数防止生成噪声
    random_state=42
)

# 执行过采样
try:
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
    print("Resampled Class Distribution:", Counter(y_tr_res))
except ValueError as e:
    print(f"Error in SMOTE: {e}")
    # 回退到自动过采样
    smote = SMOTE(sampling_strategy='auto')
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)



# 手动计算类别权重
classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_encoded
)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}

# 改进的进度条训练函数
def train_with_progress(model, X, y, total_trees=300, batch_size=30):
    """优化后的训练函数"""
    model.n_estimators = 0
    with tqdm(total=total_trees,
             desc=f"{' TRAINING PROGRESS ':#^50}",
             bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for _ in range(0, total_trees, batch_size):
            model.n_estimators += batch_size
            model.fit(X, y)
            pbar.update(batch_size)

# 模型初始化
def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,  # 限制树深度防过拟合
        min_samples_split=10,  # 增加分裂难度
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42
    )

# 第一阶段训练
print("\n" + " PHASE 1: INITIAL TRAINING (80% DATA) ".center(50, "="))
model = create_model()
train_with_progress(model, X_tr, y_tr)

# 验证集评估
val_pred = model.predict(X_val)
val_labels = label_encoder.inverse_transform(val_pred)
true_labels = label_encoder.inverse_transform(y_val)

print("\n" + " 验证报告（列名解释） ".center(50, "="))
print("| 精确率 (precision) | 召回率 (recall) | F1值 (f1-score) | 样本数 (support) |")
print("-"*65)
print(classification_report(
    true_labels, val_labels,
    target_names=label_encoder.classes_,
    digits=4
))

# 显式计算并打印关键F1指标
macro_f1 = f1_score(y_val, val_pred, average='macro')
weighted_f1 = f1_score(y_val, val_pred, average='weighted')

print(f"\n关键指标：")
print(f"宏平均F1 (所有类别同等重要): {macro_f1:.4f}")
print(f"加权平均F1 (考虑类别样本量): {weighted_f1:.4f}")
print("="*65)

print(f"\n说明：")
print("1. 精确率：预测正确的正类占所有预测为正类的比例")
print("2. 召回率：真实正类中被正确预测的比例")
print("3. F1值：精确率和召回率的调和平均数")
print("4. 样本数：该类别在验证集中的真实样本数量")
print("="*65)


# 第二阶段训练（全量数据）
print("\n" + " PHASE 2: FULL DATA TRAINING ".center(50, "="))
model_full = create_model()
train_with_progress(model_full, X_train_processed, y_train_encoded)

# 生成预测结果
test_pred = model_full.predict(X_test_processed)
test_pred_labels = label_encoder.inverse_transform(test_pred)

# 保存结果
submission = pd.DataFrame({'id': test_ids, 'attack_cat': test_pred_labels})
submission.to_csv('submission.csv', index=False)

print("\n" + " PROCESS COMPLETED ".center(50, "="))