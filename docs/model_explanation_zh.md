# State Embedding Model (model.py) 代码详解

## 概述

`src/state/emb/nn/model.py` 文件实现了 State Embedding Model（SE），这是一个基于 Transformer 的深度学习模型，用于学习单细胞基因表达数据的嵌入表示。该模型使用 PyTorch Lightning 框架构建，能够处理高维单细胞数据并生成有意义的细胞嵌入向量。

## 主要组件

### 1. SkipBlock 类（第 34-54 行）

**作用**：实现了一个带有残差连接的前馈神经网络块。

**架构**：
```
输入 X (in_features维) 
  → MLP层1: Linear(in_features → in_features*2) + ReLU
  → MLP层2: Linear(in_features*2 → in_features)
  → 残差连接: X + MLP(X)
  → LayerNorm
  → 输出
```

**参数**：
- `in_features`: 输入特征维度

**关键特性**：
- 使用残差连接（residual connection）防止梯度消失
- 使用 LayerNorm 进行归一化，提高训练稳定性
- ReLU 激活函数提供非线性变换

---

### 2. nanstd 函数（第 57-58 行）

**作用**：计算张量的标准差，忽略 NaN 值。

**实现细节**：
- 使用 `torch.nanmean` 处理缺失值
- 计算方差后取平方根得到标准差

---

### 3. StateEmbeddingModel 类（第 61-488 行）

这是核心模型类，继承自 `L.LightningModule`（PyTorch Lightning）。

#### 3.1 初始化参数（第 62-78 行）

**主要参数说明**：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `token_dim` | int | - | 输入 token 的维度（基因嵌入维度） |
| `d_model` | int | - | Transformer 模型的隐藏层维度 |
| `nhead` | int | - | 多头注意力机制的头数 |
| `d_hid` | int | - | 前馈网络的隐藏层维度 |
| `nlayers` | int | - | Transformer 编码器层数 |
| `output_dim` | int | - | 输出嵌入的维度 |
| `dropout` | float | 0.0 | Dropout 概率，用于防止过拟合 |
| `warmup_steps` | int | 0 | 学习率预热步数 |
| `compiled` | bool | False | 是否使用 PyTorch 2.0 编译优化 |
| `max_lr` | float | 4e-4 | 最大学习率 |
| `emb_cnt` | int | 145469 | 嵌入数量（未直接使用） |
| `emb_size` | int | 5120 | 嵌入大小（未直接使用） |
| `cfg` | - | None | 配置对象 |
| `collater` | - | None | 数据整理器 |

#### 3.2 模型架构组件

**a. CLS Token（第 84 行）**
```python
self.cls_token = nn.Parameter(torch.randn(1, token_dim))
```
- 可学习的分类 token，添加到每个序列的开头
- 用于聚合整个序列的信息，类似于 BERT 中的 [CLS] token

**b. 编码器（第 93-97 行）**
```python
self.encoder = nn.Sequential(
    nn.Linear(token_dim, d_model, bias=True),
    nn.LayerNorm(d_model),
    nn.SiLU(),  # Swish 激活函数
)
```
- 将输入 token 从 `token_dim` 维映射到 `d_model` 维
- 使用 SiLU（Swish）激活函数，性能优于 ReLU
- LayerNorm 在激活前进行归一化

**c. Transformer 编码器（第 99-104 行）**
```python
layers = [FlashTransformerEncoderLayer(d_model, nhead, d_hid, dropout=dropout) 
          for _ in range(nlayers)]
self.transformer_encoder = FlashTransformerEncoder(layers)
```
- 使用 Flash Attention 优化的 Transformer 层
- 堆叠 `nlayers` 个编码器层
- 每层包含多头自注意力和前馈网络

**d. 解码器（第 109-112 行）**
```python
self.decoder = nn.Sequential(
    SkipBlock(d_model),
    nn.Linear(d_model, output_dim, bias=True),
)
```
- 使用 SkipBlock 进行特征变换
- 将 Transformer 输出映射到最终的嵌入维度 `output_dim`

**e. 二分类解码器（第 121-125 行）**
```python
self.binary_decoder = nn.Sequential(
    SkipBlock(output_dim + d_model + self.z_dim),
    SkipBlock(output_dim + d_model + self.z_dim),
    nn.Linear(output_dim + d_model + self.z_dim, 1, bias=True),
)
```
- 用于预测基因表达的二分类任务（表达/不表达）
- 输入是细胞嵌入、基因嵌入和附加特征的拼接
- 使用两个 SkipBlock 进行深度特征提取

**f. 数据集校正（第 155-167 行）**
```python
if getattr(self.cfg.model, "dataset_correction", False):
    self.dataset_token = nn.Parameter(torch.randn(1, token_dim))
    self.dataset_embedder = nn.Linear(output_dim, self.z_dim_ds)
    self.dataset_encoder = nn.Sequential(...)
    self.dataset_loss = nn.CrossEntropyLoss()
```
- 可选功能，用于处理批次效应
- 添加可学习的数据集 token
- 预测样本来自哪个数据集，用于域自适应

**g. 计数编码器（第 127-133 行）**
```python
if self.cfg.model.counts:
    self.bin_encoder = nn.Embedding(10, d_model)
    self.count_encoder = nn.Sequential(
        nn.Linear(1, 512, bias=True),
        nn.LeakyReLU(),
        nn.Linear(512, 10),
    )
```
- 编码基因表达计数信息
- 使用软分箱（soft binning）方法
- 将连续的计数值映射到 10 个离散的 bin

#### 3.3 前向传播（第 288-332 行）

**输入**：
- `src`: 形状为 `[batch_size, seq_len, token_dim]` 的输入序列
- `mask`: 填充掩码
- `counts`: 基因表达计数（可选）
- `dataset_nums`: 数据集编号（可选）

**处理流程**：

1. **Token 编码**（第 295 行）
   ```python
   src = self.encoder(src) * math.sqrt(self.d_model)
   ```
   - 线性变换 + LayerNorm + SiLU
   - 乘以 √d_model 进行缩放（标准 Transformer 做法）

2. **计数嵌入**（第 296-319 行）
   - 如果提供计数信息，进行软分箱编码
   - 将计数值转换为 10 个 bin 的概率分布
   - 对 bin 嵌入进行加权求和
   - 添加到 token 嵌入中

3. **Transformer 编码**（第 321 行）
   ```python
   output = self.transformer_encoder(src, src_key_padding_mask=None)
   ```
   - 通过多层 Transformer 编码器
   - 捕获序列中的全局依赖关系

4. **解码和归一化**（第 322-325 行）
   ```python
   gene_output = self.decoder(output)
   embedding = gene_output[:, 0, :]  # 提取 CLS token
   embedding = nn.functional.normalize(embedding, dim=1)
   ```
   - 解码器生成最终表示
   - 提取 CLS token 作为细胞级嵌入
   - L2 归一化确保嵌入在单位超球面上

5. **数据集嵌入**（第 328-331 行）
   - 如果启用数据集校正，提取最后一个 token（数据集 token）
   - 用于预测样本来源

**输出**：
- `gene_output`: 所有 token 的输出表示
- `embedding`: CLS token 的归一化嵌入（细胞级表示）
- `dataset_emb`: 数据集 token 的嵌入（如果启用）

#### 3.4 训练步骤（第 363-450 行）

**shared_step 方法**：训练和验证共享的核心逻辑

1. **批次嵌入计算**（第 365 行）
   ```python
   X, Y, batch_weights, embs, dataset_embs = self._compute_embedding_for_batch(batch)
   ```
   - `X`: 基因嵌入
   - `Y`: 目标基因表达值
   - `embs`: 细胞嵌入（CLS token）
   - `dataset_embs`: 数据集嵌入

2. **特征拼接**（第 372-399 行）
   - 将细胞嵌入（CLS token）与每个基因嵌入拼接
   - 可选添加 RDA（相对深度调整）特征
   - 可选添加数据集校正特征

3. **二分类预测**（第 402 行）
   ```python
   decs = self.binary_decoder(combine)
   ```
   - 预测每个基因是否表达

4. **损失计算**（第 404-428 行）

   支持多种损失函数：
   - **Cross Entropy**（交叉熵）：二分类损失
   - **MSE**（均方误差）：回归损失
   - **Wasserstein**：分布匹配损失
   - **KL Divergence**（KL 散度）：分布相似性
   - **MMD**（最大均值差异）：核方法分布匹配
   - **Tabular**：表格数据专用损失

5. **数据集校正损失**（第 429-441 行）
   - 如果启用数据集校正，添加数据集分类损失
   - 使用交叉熵损失预测数据集标签

6. **学习率调度**（第 442-449 行）
   - 动态调整学习率
   - 支持 ReduceLROnPlateau 等策略

#### 3.5 优化器配置（第 464-483 行）

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(), 
        lr=max_lr, 
        weight_decay=self.cfg.optimizer.weight_decay
    )
    
    # 学习率调度器链
    lr_schedulers = [
        LinearLR(...),  # 线性预热
        CosineAnnealingLR(...),  # 余弦退火
    ]
    scheduler = ChainedScheduler(lr_schedulers)
```

**优化策略**：
- **AdamW 优化器**：带权重衰减的 Adam，防止过拟合
- **线性预热**：前 3% 步数从小学习率逐渐增加
- **余弦退火**：后续步数学习率按余弦曲线衰减
- **最小学习率**：max_lr * 0.3，避免学习率过小

#### 3.6 检查点保存（第 174-207 行）

**on_save_checkpoint 方法**：
- 保存配置文件的 YAML 快照
- 打包蛋白质嵌入（protein embeddings）
- 确保所有张量在 CPU 上，便于加载
- 使用异常处理，不阻塞检查点保存

#### 3.7 实用方法

**get_gene_embedding**（第 248-260 行）
- 根据基因名称获取基因嵌入
- 从预加载的蛋白质嵌入字典中查找
- 对缺失的基因使用零向量

**resize_batch**（第 262-286 行）
- 静态方法，用于调整批次大小
- 将细胞嵌入和任务嵌入组合成笛卡尔积
- 支持添加计数和数据集嵌入

**_log_nonzero_elements_stats**（第 334-361 行）
- 记录非零元素统计信息
- 用于消融研究（ablation study）
- 分析填充对模型学习的影响

---

## 模型架构的意义

### 1. 为什么使用 Transformer？

- **长程依赖**：单细胞数据中，基因之间存在复杂的调控网络，Transformer 的自注意力机制能够捕获这些长程依赖关系
- **并行计算**：相比 RNN，Transformer 可以并行处理整个序列，训练速度更快
- **可解释性**：注意力权重可以揭示基因之间的相互作用

### 2. CLS Token 的作用

- **序列级表示**：CLS token 通过自注意力聚合整个基因表达序列的信息
- **下游任务**：归一化的 CLS 嵌入可用于细胞聚类、分类、检索等任务
- **类似 BERT**：借鉴了 BERT 的设计，在单细胞领域表现良好

### 3. 软分箱（Soft Binning）

- **离散化表达**：将连续的基因表达计数离散化为 bin
- **平滑表示**：使用概率分布而非硬分配，保留更多信息
- **灵感来源**：借鉴 scFoundation 模型的设计

### 4. 数据集校正

- **批次效应**：不同实验批次可能引入系统性偏差
- **域自适应**：通过预测数据集来源，学习不变特征
- **多任务学习**：主任务（基因表达预测）+ 辅助任务（数据集分类）

### 5. 残差连接和归一化

- **训练稳定性**：SkipBlock 使用残差连接，缓解梯度消失
- **特征归一化**：LayerNorm 和 L2 归一化确保特征分布稳定
- **深层网络**：支持堆叠更多层，提升模型容量

### 6. 多样的损失函数

- **任务适配**：不同损失函数适用于不同的下游任务
- **分布匹配**：Wasserstein、MMD 等损失关注整体分布，适合生成任务
- **分类任务**：交叉熵用于二分类（基因表达/不表达）

---

## 训练流程总结

1. **输入准备**：
   - 基因序列（token）
   - 基因表达计数（可选）
   - 数据集标签（可选）

2. **前向传播**：
   - Token 编码 → Transformer → 解码 → CLS 嵌入
   - 基因嵌入 × 细胞嵌入 → 二分类预测

3. **损失计算**：
   - 主损失：基因表达预测损失
   - 辅助损失：数据集分类损失（可选）

4. **反向传播和优化**：
   - AdamW 优化器更新参数
   - 学习率调度器动态调整学习率

5. **验证和评估**：
   - 共享 shared_step 逻辑
   - 记录验证损失和统计信息

---

## 使用建议

1. **参数调优**：
   - `d_model` 和 `output_dim`：增大以提升模型容量，但会增加计算成本
   - `nlayers` 和 `nhead`：平衡深度和宽度
   - `dropout`：根据数据集大小调整，防止过拟合

2. **训练技巧**：
   - 使用 `compiled=True` 开启 PyTorch 2.0 编译优化
   - 调整 `warmup_steps` 和学习率以稳定训练
   - 监控 `avg_nonzero_genes` 等指标

3. **扩展性**：
   - 代码支持多种损失函数和数据集校正
   - 可以轻松添加新的编码器或解码器组件

---

## 关键技术点

- **Flash Attention**：优化的注意力实现，减少内存占用和计算时间
- **PyTorch Lightning**：简化训练循环，自动处理分布式训练、检查点等
- **动态配置**：使用 OmegaConf 管理配置，灵活性高
- **检查点持久化**：保存配置和嵌入，便于推理和迁移学习

---

## 总结

`StateEmbeddingModel` 是一个复杂而强大的单细胞基因表达嵌入模型，结合了：
- Transformer 的强大序列建模能力
- 领域特定的设计（软分箱、数据集校正）
- 灵活的损失函数和优化策略
- 工程化的最佳实践（编译优化、检查点管理）

该模型适用于单细胞数据的嵌入学习、细胞类型注释、批次效应校正等多种任务。
