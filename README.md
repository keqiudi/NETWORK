# 网络流量安全事件分类模型

本项目基于 Python，集成 XGBoost 和 RandomForest，实现对网络流量攻击类别的高精度识别，适配多种不平衡样本场景。特征工程特别针对 Analysis、Backdoor、DoS、Worms 等弱类增强，具备强泛化能力。支持采样、类别权重调整、概率融合与阈值优化。

## 环境依赖

请先安装依赖：

```bash
pip install -r requirements.txt
```

## 数据要求

- **训练集**：`train_data.csv`，需含 `id`、`attack_cat` 及原始特征列
- **测试集**：`test_data.csv`，需含 `id` 及原始特征列（无 attack_cat）

## 运行方式

```bash
python your_main_script.py
```
*确保 `train_data.csv`、`test_data.csv` 与脚本在同一目录下。*

### 主要流程

1. **数据加载与特征工程**  
   - 动态低频类别合并
   - 组合交互特征、分桶、对数变换、极端标志
2. **编码**  
   - OneHotEncoder 处理类别特征
   - LabelEncoder 处理目标变量
3. **采样与权重调整**  
   - 使用 ADASYN 对 Analysis/Backdoor/DoS/Worms 等弱类过采样
   - 类别权重提升弱类贡献
4. **模型融合**  
   - XGBoost 与 RandomForest 融合，针对弱类概率加权
   - 支持概率阈值优化，进一步提升小类召回率
5. **模型评估与测试集预测**  
   - 输出详细 classification_report
   - 生成 `submission.csv`，格式为 id, attack_cat

## 数据增强说明

- **过采样策略**：采用 ADASYN 对弱类样本进行智能过采样，目标为主类样本数的 40%，显著提升小类的学习能力和召回率。
- **类别权重调整**：对 Analysis/Backdoor/Worms 等类提升损失权重，进一步平衡学习。

## 参数配置与调整建议

- 支持手动与自动（Optuna）调参，建议先手动粗调后自动细调。
- 推荐多折交叉验证，优选宏 F1 高且稳定的参数组合，防止过拟合本地验证集。

## 结果输出

- `submission.csv`：最终测试集预测结果
- 控制台输出训练集、测试集类别分布与详细 F1 指标

## 复现及自定义

如需更换采样方法、模型结构、特征工程细节，可直接修改对应代码区块。  
如遇依赖/兼容性等问题，优先升级至上述 required 版本。

