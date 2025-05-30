# ==================== 0. 环境设置 ====================
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from imblearn.over_sampling import ADASYN
from collections import Counter
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

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
def advanced_feature_engineering(df):
    """精准匹配字段的安全事件特征工程，特别针对Analysis、Backdoor、DoS、Worms类别增强"""

    # ---- 通用基础特征与交互 ----
    df['proto_service'] = df['proto'] + '_' + df['service']
    df['proto_state'] = df['proto'] + '_' + df['state']
    df['dbytes_sbytes_ratio'] = df['dbytes'] / (df['sbytes'] + 1e-5)
    df['spkts_dpkts_ratio'] = df['spkts'] / (df['dpkts'] + 1e-5)
    df['total_bytes'] = df['sbytes'] + df['dbytes']
    df['bytes_per_pkt'] = df['total_bytes'] / (df['spkts'] + df['dpkts'] + 1e-5)
    df['spkts_per_sec'] = df['spkts'] / (df['dur'] + 1e-5)
    df['dpkts_per_sec'] = df['dpkts'] / (df['dur'] + 1e-5)

    # ---- log变换特征 ----
    for col in ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur']:
        df[f'log_{col}'] = np.log1p(df[col])

    # ---- 分桶特征 ----
    for col in ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur']:
        df[f'{col}_bin'] = pd.qcut(df[col], q=5, duplicates='drop', labels=False)

    # ---- 离群/极端流量标志 ----
    df['is_large_duration'] = (df['dur'] > df['dur'].quantile(0.98)).astype(int)
    df['is_large_bytes'] = (df['sbytes'] > df['sbytes'].quantile(0.98)).astype(int)
    df['is_many_pkts'] = (df['spkts'] > df['spkts'].quantile(0.98)).astype(int)

    # ---- 稀有协议与服务 ----
    df['is_rare_proto'] = df['proto'].isin(df['proto'].value_counts()[df['proto'].value_counts() < 10].index).astype(
        int)
    df['is_rare_service'] = df['service'].isin(
        df['service'].value_counts()[df['service'].value_counts() < 10].index).astype(int)

    # ---- Analysis特征 ----
    df['is_analysis_proto'] = df['proto'].isin(['unas', 'ospf', 'sctp', 'qnx', 'wsn']).astype(int)
    df['is_analysis_service'] = df['service'].isin(['-', 'smtp', 'dns']).astype(int)
    df['is_int_state'] = (df['state'] == 'INT').astype(int)
    df['is_small_analysis'] = ((df['spkts'] <= 2) & (df['sbytes'] < 220)).astype(int)
    df['analysis_long_conn'] = (df['dur'] > 10).astype(int)

    # ---- Backdoor特征 ----
    df['is_backdoor_proto'] = df['proto'].isin(['unas', 'sctp', 'tcp']).astype(int)
    df['is_backdoor_state'] = (df['state'] == 'INT').astype(int)
    df['is_short_backdoor'] = (df['dur'] < 0.0001).astype(int)
    df['is_small_backdoor'] = ((df['spkts'] <= 2) & (df['sbytes'] < 220)).astype(int)

    # ---- DoS特征 ----
    df['is_dos_proto'] = df['proto'].isin(['ospf', 'unas', 'tcp', 'udp']).astype(int)
    df['is_dos_state'] = df['state'].isin(['INT', 'CON']).astype(int)
    df['is_high_rate'] = (df.get('rate', 0) > 10000).astype(int)
    df['pkt_rate'] = df['spkts'] / (df['dur'] + 1e-5)
    df['bytes_rate'] = df['sbytes'] / (df['dur'] + 1e-5)
    df['is_long_dos'] = (df['dur'] > 10).astype(int)
    df['is_very_high_pkt'] = (df['spkts'] > 100).astype(int)
    df['is_very_high_bytes'] = (df['sbytes'] > 10000).astype(int)

    # ---- Worms特征 ----
    df['is_worm_proto'] = df['proto'].isin(['unas', 'tcp']).astype(int)
    df['is_worm_state'] = df['state'].isin(['INT', 'CON']).astype(int)
    df['is_worm_small'] = ((df['spkts'] <= 2) & (df['sbytes'] < 220)).astype(int)
    df['is_short_worm'] = (df['dur'] < 0.0001).astype(int)

    # ---- 其他有效特征 ----
    df['tcp_win_anomaly'] = ((df['swin'] < 64) | (df['dwin'] < 64)).astype(int)
    df['response_ratio'] = df['dbytes'] / (df['sbytes'] + 1e-5)

    return df

# 应用特征工程
X_train = advanced_feature_engineering(X_train)
X_test = advanced_feature_engineering(X_test)


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

# 需要编码的自造类别特征
combo_features = []
for col in ['proto_service', 'proto_state']:
    if col in X_train_processed.columns:
        combo_features.append(col)

for col in combo_features:
    le = LabelEncoder()
    # 先转成str，避免有nan或其他类型
    X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
    X_test_processed[col] = le.transform(X_test_processed[col].astype(str))

# 检查特征工程生成的非数值列
if 'proto_service' in X_train_processed.columns:
    le = LabelEncoder()
    X_train_processed['proto_service'] = le.fit_transform(X_train_processed['proto_service'].astype(str))
    X_test_processed['proto_service'] = le.transform(X_test_processed['proto_service'].astype(str))

# 确保所有列为数值型
print(X_train_processed.dtypes)

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

# 重置索引以避免索引不一致问题
X_tr = X_tr.reset_index(drop=True)
y_tr = pd.Series(y_tr).reset_index(drop=True)


# ==================== 5.2 过采样配置（强烈建议开启！）====================


# 定义弱势类别
weak_classes = ['Analysis', 'Backdoor', 'DoS', 'Worms']
weak_indices = [i for i, name in enumerate(label_encoder.classes_) if name in weak_classes]

# 计算原始分布
original_counts = Counter(y_tr)
# 采样策略：小类样本数扩充到主流类的40%（可微调）
max_major = max(original_counts[i] for i in set(y_tr) if i not in weak_indices)
sampling_strategy = {i: int(max_major * 0.4) for i in weak_indices}

adasyn = ADASYN(
    sampling_strategy=sampling_strategy,
    n_neighbors=5,
    random_state=42
)

try:
    X_tr_res, y_tr_res = adasyn.fit_resample(X_tr, y_tr)
    print(f"过采样后样本分布: {pd.Series(y_tr_res).value_counts()}")
except Exception as e:
    print(f"过采样失败: {e}")
    X_tr_res, y_tr_res = X_tr.copy(), y_tr.copy()

# ==================== 6. 模型配置 ====================
# 6.1 类别权重计算
classes = np.unique(y_train_encoded)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train_encoded
)

class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
# 手动提高小类别权重
difficult_classes = ['Analysis', 'Backdoor', 'Worms']
difficult_class_indices = [i for i, name in enumerate(label_encoder.classes_) if name in difficult_classes]

for cls_idx in difficult_class_indices:
    class_weight_dict[cls_idx] *= 2.5  # 增加小类别权重倍数

print("调整后的类别权重:", class_weight_dict)


# =============== 1. 参数开关和手动参数 ===============
use_search = False  # True=自动搜索, False=用手动参数

manual_rf_params = {
    'n_estimators': 343,
    'max_depth': 27, # already best
    'min_samples_split': 6, # best: 4 或 6
    'class_weight': 'balanced' # best
}
manual_xgb_params = {
    'n_estimators': 157, # 130 F1:0.54027
    'max_depth': 7, # already best
    'learning_rate': 0.1485827437659959,
    'subsample': 0.6742557566476113,
    'colsample_bytree': 0.702896135755186,
}

# ==================== 数据采样与划分 ====================
X_sample = X_tr_res.sample(frac=0.3, random_state=42)
y_sample = y_tr_res[X_sample.index]

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

if use_search:
    # ==================== 随机森林目标函数 ====================
    def optimize_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 350),
            'max_depth': trial.suggest_int('max_depth', 27, 27),
            'min_samples_split': trial.suggest_int('min_samples_split', 6,6),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
        }
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        rf.fit(X_train_sub, y_train_sub)
        y_pred = rf.predict(X_val_sub)
        return f1_score(y_val_sub, y_pred, average='macro')

    # ==================== XGBoost目标函数 ====================
    def optimize_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 350),
            'max_depth': trial.suggest_int('max_depth', 7, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        xgb = XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            tree_method='hist',
            n_jobs= 1,
            **params
        )
        xgb.fit(X_train_sub, y_train_sub)
        y_pred = xgb.predict(X_val_sub)
        return f1_score(y_val_sub, y_pred, average='macro')

    # ==================== 并行超参数优化 ====================
    print("\nOptimizing RandomForest Hyperparameters...")
    rf_study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    rf_study.optimize(optimize_rf, n_trials=20, n_jobs=10, timeout=60)

    print("RandomForest 最佳参数:", rf_study.best_params)
    print("RandomForest 最佳 F1 分数:", rf_study.best_value)

    print("\nOptimizing XGBoost Hyperparameters...")
    xgb_study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    xgb_study.optimize(optimize_xgb, n_trials=20, n_jobs=10, timeout=60)

    print("XGBoost 最佳参数:", xgb_study.best_params)
    print("XGBoost 最佳 F1 分数:", xgb_study.best_value)

    best_rf_params = rf_study.best_params
    best_xgb_params = xgb_study.best_params
else:
    print("使用手动指定参数")
    best_rf_params = manual_rf_params
    best_xgb_params = manual_xgb_params

# ==================== 6.3 模型初始化器 ====================
def create_model():
    rf = RandomForestClassifier(
        **best_rf_params,
        n_jobs=-1,
        random_state=42,
    )

    xgb = XGBClassifier(
        **best_xgb_params,
        tree_method='hist',
        random_state=42,
    )

    small_class_names = ['Analysis', 'Backdoor', 'DoS', 'Worms']
    small_class_idx = [label_encoder.transform([name])[0] for name in small_class_names if name in label_encoder.classes_]

    class HybridModel:
        def __init__(self):
            self.rf = rf
            self.xgb = xgb

        def fit(self, X, y):
            self.rf.fit(X, y)
            self.xgb.fit(X, y)

        def predict(self, X):
            rf_pred = self.rf.predict_proba(X)
            xgb_pred = self.xgb.predict_proba(X)
            weights = np.ones_like(rf_pred[0])
            for idx in small_class_idx:
                weights[idx] = 1.86
            total_pred = (0.7 * xgb_pred + 0.3 * rf_pred) * weights
            return np.argmax(total_pred, axis=1)

        def predict_proba(self, X):
            rf_pred = self.rf.predict_proba(X)
            xgb_pred = self.xgb.predict_proba(X)
            weights = np.ones_like(rf_pred[0])
            for idx in small_class_idx:
                weights[idx] = 1.86
            return (0.7 * xgb_pred + 0.3 * rf_pred) * weights

    return HybridModel()

# ==================== 阈值优化 ====================
from sklearn.metrics import precision_recall_curve

def apply_optimized_thresholds(model, X_val, small_class_names, base_threshold=0.5, aggressive_threshold=0.10):
    y_proba = model.predict_proba(X_val)
    final_pred = np.argmax(y_proba, axis=1)
    for cls in small_class_names:
        if cls in label_encoder.classes_:
            idx = label_encoder.transform([cls])[0]
            class_proba = y_proba[:, idx]
            final_pred[class_proba >= aggressive_threshold] = idx
    return final_pred

# ==================== 7. 模型训练与验证 ====================

# 带进度条的训练函数
def train_with_progress(model, X, y, X_val=None, y_val=None, total_trees=300, batch_size=30):
    """带验证和训练集监控的训练"""
    model.rf.n_estimators = 0  # 仅监控随机森林部分
    best_score = 0

    with tqdm(total=total_trees,
              desc=f"{' TRAINING PROGRESS ':#^50}",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for i in range(0, total_trees, batch_size):
            model.rf.n_estimators += batch_size
            model.fit(X, y)

            # 每50棵树验证一次
            if X_val is not None and (i // batch_size) % 5 == 0:
                # 训练集F1
                train_pred = model.predict(X)
                train_f1 = f1_score(y, train_pred, average='macro')
                # 验证集F1
                val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, val_pred, average='macro')
                pbar.set_postfix({
                    'Train F1': f'{train_f1:.4f}',
                    'Val F1': f'{val_f1:.4f}'
                })
                if val_f1 > best_score:
                    best_score = val_f1

            pbar.update(batch_size)
    print(f"Best Val F1: {best_score:.4f}")


# 7.1 第一阶段训练
print("\n" + " PHASE 1: INITIAL TRAINING (80%训练 20%验证) ".center(50, "="))

model = create_model()
train_with_progress(model, X_tr_res, y_tr_res, X_val, y_val)

# 7.2 验证集评估（包含阈值优化结果）
val_pred = model.predict(X_val)
# 激进阈值优化
small_class_names = ['Analysis', 'Backdoor', 'DoS', 'Worms']
val_pred_opt = apply_optimized_thresholds(model, X_val, small_class_names, base_threshold=0.5, aggressive_threshold=0.18)
val_labels = label_encoder.inverse_transform(val_pred)
val_labels_opt = label_encoder.inverse_transform(val_pred_opt)
true_labels = label_encoder.inverse_transform(y_val)

print("\n" + " 验证报告 ".center(50, "="))
print(classification_report(true_labels, val_labels, target_names=label_encoder.classes_, digits=4))
print("\n" + " 阈值优化后验证报告 ".center(50, "="))
print(classification_report(true_labels, val_labels_opt, target_names=label_encoder.classes_, digits=4))

macro_f1 = f1_score(y_val, val_pred, average='macro')
macro_f1_opt = f1_score(y_val, val_pred_opt, average='macro')
weighted_f1 = f1_score(y_val, val_pred, average='weighted')
weighted_f1_opt = f1_score(y_val, val_pred_opt, average='weighted')

print(f"\n核心指标：")
print(f"► 宏平均F1: {macro_f1:.4f}（普通） | {macro_f1_opt:.4f}（阈值优化）")
print(f"► 加权平均F1: {weighted_f1:.4f} | {weighted_f1_opt:.4f}")
print("=" * 65)



# ==================== 8. 全量训练与结果生成 ====================
# 8.1 第二阶段训练
print("\n" + " PHASE 2: FULL DATA TRAINING ".center(50, "="))

model_full = create_model()  # 创建新的 HybridModel 实例
train_with_progress(model_full, X_train_processed, y_train_encoded)  # 使用全量数据训练

try:
    check_is_fitted(model_full.rf)
    print("RandomForestClassifier 已被成功训练。")
except NotFittedError:
    print("RandomForestClassifier 尚未训练！")

try:
    check_is_fitted(model_full.xgb)
    print("XGBoost 已被成功训练。")
except NotFittedError:
    print("XGBoost 尚未训练！")

# 8.2 测试集预测
print("\n" + " TEST SET PREDICTION ".center(50, "="))
assert X_test_processed.shape[1] == X_train_processed.shape[1], "测试集和训练集特征数不一致！"

test_pred = model_full.predict(X_test_processed)
test_pred_labels = label_encoder.inverse_transform(test_pred)

# 8.3 保存结果
submission = pd.DataFrame({'id': test_ids, 'attack_cat': test_pred_labels})
submission.to_csv('submission.csv', index=False)

print("\n" + " PROCESS COMPLETED ".center(50, "="))

print("训练集类别分布:")
print(pd.Series(y_train).value_counts(normalize=True))

print("测试集类别分布:")
print(submission['attack_cat'].value_counts(normalize=True))