# ==================== 0. 环境设置 ====================
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

# ==================== 1. 数据加载与预处理 ====================
# 1.1 加载原始数据
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# 1.2 特征/标签分离
X_train = train_data.drop(columns=['id', 'attack_cat'])
y_train = train_data['attack_cat']
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

# ==================== 2. 类别特征处理 ====================
# 2.1 定义初始类别列
categorical_cols = ['proto', 'service', 'state']

# 2.2 动态低频合并
for col in categorical_cols:
    freq = X_train[col].value_counts()
    # 计算累积分布
    cum_ratio = freq.cumsum() / freq.sum()
    # 保留覆盖95%样本的类别
    threshold = freq[cum_ratio < 0.95].min()

    # 应用合并规则
    X_train[col] = X_train[col].apply(lambda x: x if freq[x] >= threshold else 'other')
    X_test[col] = X_test[col].apply(lambda x: x if x in freq and freq[x] >= threshold else 'other')

# ==================== 3. 特征工程 ====================
# 3.1 创建协议-服务组合特征
for df in [X_train, X_test]:
    df['proto_service'] = df['proto'] + '_' + df['service']
    
# 3.2 更新类别列定义
categorical_cols = ['proto', 'service', 'state', 'proto_service']

# ==================== 4. 数据编码 ====================
# 4.1 OneHot编码器初始化
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[categorical_cols])


# 4.2 编码转换函数
def encode_features(df, encoder):
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded,
                              columns=encoder.get_feature_names_out(categorical_cols))
    return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)


# 4.3 执行编码
X_train_processed = encode_features(X_train, encoder)
X_test_processed = encode_features(X_test, encoder)

# 4.4 标签编码
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# ==================== 5. 验证集准备 ====================
# 5.1 分层划分验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train_encoded,
    test_size=0.2,
    stratify=y_train_encoded,
    random_state=42
)

# 5.2 过采样配置
from imblearn.over_sampling import SMOTE
from collections import Counter

# 定义弱势类别
weak_classes = ['Analysis', 'Backdoor', 'DoS', 'Worms']
weak_indices = [i for i, name in enumerate(label_encoder.classes_) if name in weak_classes]

# 计算原始分布
original_counts = Counter(y_tr)

# 动态采样策略
sampling_strategy = {
    i: max(int(1.5 * original_counts[i]), original_counts[i] + 1)
    for i in weak_indices
}

# 5.3 执行过采样
smote = SMOTE(
    sampling_strategy=sampling_strategy,
    k_neighbors=5,
    random_state=42
)

try:
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
except ValueError as e:
    print(f"过采样异常: {e}, 回退到自动模式")
    smote = SMOTE(sampling_strategy='auto')
    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

# ==================== 6. 模型配置 ====================
# 6.1 类别权重计算
classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_encoded
)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}


# 6.2 带进度条的训练函数
def train_with_progress(model, X, y, total_trees=300, batch_size=30):
    model.n_estimators = 0
    with tqdm(total=total_trees,
              desc=f"{' TRAINING PROGRESS ':#^50}",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for _ in range(0, total_trees, batch_size):
            model.n_estimators += batch_size
            model.fit(X, y)
            pbar.update(batch_size)


# 6.3 模型初始化器
def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        class_weight=class_weight_dict,
        n_jobs=-1,
        random_state=42
    )


# ==================== 7. 模型训练与验证 ====================
# 7.1 第一阶段训练
print("\n" + " PHASE 1: INITIAL TRAINING (80% DATA) ".center(50, "="))
model = create_model()
train_with_progress(model, X_tr_res, y_tr_res)

# 7.2 验证集评估
val_pred = model.predict(X_val)
val_labels = label_encoder.inverse_transform(val_pred)
true_labels = label_encoder.inverse_transform(y_val)

# 7.3 生成报告
print("\n" + " 验证报告 ".center(50, "="))
print(classification_report(
    true_labels, val_labels,
    target_names=label_encoder.classes_,
    digits=4
))

# 7.4 关键指标计算
macro_f1 = f1_score(y_val, val_pred, average='macro')
weighted_f1 = f1_score(y_val, val_pred, average='weighted')

print(f"\n核心指标：")
print(f"► 宏平均F1: {macro_f1:.4f}")
print(f"► 加权平均F1: {weighted_f1:.4f}")
print("=" * 65)

# ==================== 8. 全量训练与结果生成 ====================
# 8.1 第二阶段训练
print("\n" + " PHASE 2: FULL DATA TRAINING ".center(50, "="))
model_full = create_model()
train_with_progress(model_full, X_train_processed, y_train_encoded)

# 8.2 测试集预测
test_pred = model_full.predict(X_test_processed)
test_pred_labels = label_encoder.inverse_transform(test_pred)

# 8.3 保存结果
submission = pd.DataFrame({'id': test_ids, 'attack_cat': test_pred_labels})
submission.to_csv('submission.csv', index=False)

print("\n" + " PROCESS COMPLETED ".center(50, "="))