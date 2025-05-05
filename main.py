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
    """精准匹配字段的安全事件特征工程"""
    # Backdoor特征
    df['is_unas_proto'] = (df['proto'] == 'unas').astype(int)
    df['long_duration'] = (df['dur'] > 1).astype(int)  # 增加长连接特征

    # DoS特征（使用连接时间统计替代端口）
    df['is_dos_proto'] = df['proto'].isin(['ospf', 'ggp']).astype(int)
    df['high_traffic'] = ((df['sbytes'] > 1000) | (df['spkts'] > 40)).astype(int)
    df['long_duration_conn'] = (df['ct_dst_sport_ltm'] > 60).astype(int)  # 长连接标记

    # Analysis特征（通过服务类型判断）
    df['is_dns_service'] = (df['service'] == 'dns').astype(int)
    df['small_packet'] = ((df['spkts'] <= 2) & (df['sbytes'] < 200)).astype(int)
    df['short_duration'] = (df['dur'] < 0.0001).astype(int)

    # 协议-服务组合特征
    df['proto_service'] = df['proto'] + '_' + df['service']

    # 其他有效特征
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

# 5.2 过采样配置
from imblearn.over_sampling import SMOTE
from collections import Counter
from imblearn.over_sampling import ADASYN

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
# 使用ADASYN代替SMOTE
adasyn = ADASYN(
    sampling_strategy=sampling_strategy,
    n_neighbors=5,
    random_state=42
)

# 添加少数类样本生成监控
try:
    X_tr_res, y_tr_res = adasyn.fit_resample(X_tr, y_tr)
    print(f"生成样本分布: {pd.Series(y_tr_res).value_counts()}")
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
    class_weight_dict[cls_idx] *= 2.0  # 增加小类别权重倍数

print("调整后的类别权重:", class_weight_dict)



# 6.2 超参数搜索
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ==================== 数据采样与划分 ====================
# 抽取 30% 数据用于调优
X_sample = X_tr_res.sample(frac=0.3, random_state=42)
y_sample = y_tr_res[X_sample.index]

# 固定验证集
X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

# ==================== 随机森林目标函数 ====================
def optimize_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'max_depth': trial.suggest_int('max_depth', 15, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    rf.fit(X_train_sub, y_train_sub)
    y_pred = rf.predict(X_val_sub)
    return f1_score(y_val_sub, y_pred, average='macro')

# ==================== XGBoost目标函数 ====================
def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 350),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    xgb = XGBClassifier(
        random_state=42,
        objective='multi:softprob',
        tree_method='hist',  # 高效直方图方法
        n_jobs=-1,
        **params
    )
    xgb.fit(X_train_sub, y_train_sub)
    y_pred = xgb.predict(X_val_sub)
    return f1_score(y_val_sub, y_pred, average='macro')

# ==================== 并行超参数优化 ====================
# 随机森林调优
print("\nOptimizing RandomForest Hyperparameters...")
rf_study = optuna.create_study(direction="maximize", pruner=MedianPruner())
rf_study.optimize(optimize_rf, n_trials=50, n_jobs=10, timeout=60)  # 允许并行运行 10 个线程，限制 60 秒

print("RandomForest 最佳参数:", rf_study.best_params)
print("RandomForest 最佳 F1 分数:", rf_study.best_value)

# XGBoost调优
print("\nOptimizing XGBoost Hyperparameters...")
xgb_study = optuna.create_study(direction="maximize", pruner=MedianPruner())
xgb_study.optimize(optimize_xgb, n_trials=50, n_jobs=10, timeout=60)  # 允许并行运行 10 个线程，限制 60 秒

print("XGBoost 最佳参数:", xgb_study.best_params)
print("XGBoost 最佳 F1 分数:", xgb_study.best_value)


# 6.3 模型初始化器
def create_model():


    # 保留原有随机森林配置
    rf = RandomForestClassifier(
        **rf_study.best_params,
        n_jobs=-1,
        random_state=42
    )

    # 新增XGBoost模型集成
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        **xgb_study.best_params,
        tree_method='hist',
        random_state=42
    )

    # 创建混合模型类
    class HybridModel:
        def __init__(self):
            self.rf = rf
            self.xgb = xgb

        def fit(self, X, y):
            self.rf.fit(X, y)  # 训练 RandomForestClassifier
            self.xgb.fit(X, y)  # 训练 XGBoost

        def predict(self, X):
            rf_pred = self.rf.predict_proba(X)
            xgb_pred = self.xgb.predict_proba(X)
            # 加权融合预测结果
            return np.argmax(0.6 * xgb_pred + 0.4 * rf_pred, axis=1)

        def predict_proba(self, X):
            rf_pred = self.rf.predict_proba(X)
            xgb_pred = self.xgb.predict_proba(X)
            # 加权融合概率
            return 0.6 * xgb_pred + 0.4 * rf_pred

    return HybridModel()

# ==================== 7. 模型训练与验证 ====================

# 带进度条的训练函数
def train_with_progress(model, X, y, X_val=None, y_val=None, total_trees=300, batch_size=30):
    """带验证监控的训练"""
    model.rf.n_estimators = 0  # 仅监控随机森林部分
    best_score = 0

    with tqdm(total=total_trees,
              desc=f"{' TRAINING PROGRESS ':#^50}",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for _ in range(0, total_trees, batch_size):
            model.rf.n_estimators += batch_size
            model.fit(X, y)

            # 每50棵树验证一次
            if X_val is not None and (_ // batch_size) % 5 == 0:
                current_pred = model.predict(X_val)
                current_f1 = f1_score(y_val, current_pred, average='macro')
                pbar.set_postfix({'Val F1': f'{current_f1:.4f}'})
                if current_f1 > best_score:
                    best_score = current_f1

            pbar.update(batch_size)


# 7.1 第一阶段训练
print("\n" + " PHASE 1: INITIAL TRAINING (80%训练 20%验证) ".center(50, "="))

model = create_model()
train_with_progress(model, X_tr_res, y_tr_res, X_val, y_val)  # 传入验证数据

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

# 在现有代码后添加阈值优化
from sklearn.metrics import precision_recall_curve

def optimize_threshold(model, X_val, y_val, target_class):
    """为指定类别优化预测阈值"""
    class_idx = label_encoder.transform([target_class])[0]
    y_proba = model.predict_proba(X_val)[:, class_idx]  # 获取目标类别的概率

    precision, recall, thresholds = precision_recall_curve(
        (y_val == class_idx).astype(int), y_proba
    )

    # 寻找满足最低召回率的阈值
    viable_thresholds = thresholds[recall[:-1] > 0.3]  # 设定最低召回率为 30%
    if len(viable_thresholds) > 0:
        best_threshold = viable_thresholds[np.argmax(precision[:-1][recall[:-1] > 0.3])]
    else:
        best_threshold = 0.5

    return best_threshold


# 针对多个目标类别优化阈值
target_classes = ['Analysis', 'Backdoor', 'DoS']
optimized_thresholds = {}

for target_class in target_classes:
    optimized_thresholds[target_class] = optimize_threshold(model, X_val, y_val, target_class)
    print(f"优化后的 {target_class} 类别阈值: {optimized_thresholds[target_class]:.4f}")

# 应用优化阈值
def apply_optimized_thresholds(model, X_val, optimized_thresholds, target_classes):
    """根据优化后的阈值调整预测结果"""
    y_proba = model.predict_proba(X_val)
    final_pred = np.argmax(y_proba, axis=1)  # 默认预测结果

    for target_class in target_classes:
        class_idx = label_encoder.transform([target_class])[0]
        threshold = optimized_thresholds[target_class]

        # 根据阈值调整预测
        class_proba = y_proba[:, class_idx]
        final_pred[(class_proba >= threshold)] = class_idx

    return final_pred


# 使用优化后的阈值调整预测结果
val_pred_optimized = apply_optimized_thresholds(model, X_val, optimized_thresholds, target_classes)
val_labels_optimized = label_encoder.inverse_transform(val_pred_optimized)

# 生成优化后的报告
print("\n" + " 优化后验证报告 ".center(50, "="))
print(classification_report(
    true_labels, val_labels_optimized,
    target_names=label_encoder.classes_,
    digits=4
))

# ==================== 8. 全量训练与结果生成 ====================
# 8.1 第二阶段训练
print("\n" + " PHASE 2: FULL DATA TRAINING ".center(50, "="))

model_full = create_model()  # 创建新的 HybridModel 实例
train_with_progress(model_full, X_train_processed, y_train_encoded)  # 使用全量数据训练

# 验证模型是否正确训练
from sklearn.utils.validation import check_is_fitted

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