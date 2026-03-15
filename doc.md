

# Motif-Aware VQ-MoE (基于子结构量化的动态专家模型)

conda activate /ibex/user/songt/conda_envs/momo
---

## Step 1: 数据预处理 (Data Preprocessing)

**目标**：将 PCQM4Mv2 的数据转化为“原子-Motif”层级结构，并提取局部 3D 标签。

### 1.1 数据源
*   **预训练**: [PCQM4Mv2 (OGB)] data/pcqm4m-v2-train.sdf
*   **微调**: [MoleculeNet](https://moleculenet.org/) - 仅 2D SMILES。

### 1.2 处理流程 (必须在 DataLoader 之外预先处理好)
你需要编写脚本，对每个分子执行以下操作：

1.  **Motif 分解**:
    *   使用 `rdkit.Chem.BRICS.BreakBRICSBonds` 切割分子。
    *   **Mapping**: 建立一个索引数组 `batch.motif_id`（注意：不是 `motif_batch`，以避免与 PyG 的 `*_batch` 合并语义冲突），长度等于原子数。例如：原子 0-5 属于 Motif A (索引 0)，原子 6-8 属于 Motif B (索引 1)。
2.  **构建局部 3D 标签 (Ground Truth)**:
    *   **不要使用全局坐标**（因为旋转平移不具备不变性）。
    *   对于每个 Motif，计算其**内部几何特征向量 $z_{3D}^{GT}$**：
        *   包含：键长均值/方差、键角均值/方差、以及惯性主轴的特征值（描述形状是扁的还是长的）。
    *   这个 $z_{3D}^{GT}$ 将作为预训练时 MLP Expert 想要拟合的目标。

---

## Step 2: 模型架构 (Model Architecture)

模型由 **原子编码器** $\rightarrow$ **Motif 聚合层** $\rightarrow$ **VQ-MoE 层** $\rightarrow$ **读出层** 组成。

### 2.1 2D Encoder (Hierarchical GNN)
*   **第一层**: 5层 **GIN (Graph Isomorphism Network)**。
    *   输入：原子特征。
    *   输出：原子 Embedding ($H_{atom}$).
*   **第二层**: **Motif Pooling** (关键步骤)。
    *   操作：基于 `motif_id` 索引，对属于同一个 Motif 的原子 Embedding 求均值 (Mean Pooling)。
    *   输出：**Motif Embedding ($H_{motif}^{2D}$)**。

### 2.2 3D Encoder (Teacher - 仅预训练)
*   使用 **SchNet** 处理 3D 坐标，同样进行 Motif Pooling，得到 **3D Motif Embedding ($z_{motif}^{3D}$)**。
*   *注意*：这个向量仅用于监督 VQ Codebook 的聚类中心。

### 2.3 Motif-VQ-MoE Layer (核心创新)
这是插入在 Motif Pooling 之后的部分：

1.  **Router (分类器)**:
    *   输入：$H_{motif}^{2D}$。
    *   输出：概率分布 $P(k)$，选择 Top-1 或 Top-K 的 **Codebook Index $k$**。
    *   *物理意义*：识别这个 Motif 是“苯环类”还是“长链类”。
2.  **Codebook (静态原型)**:
    *   存储 $K$ 个向量 $E_k$。每个向量代表一种**标准几何构象**的中心。
3.  **Expert (动态 MLP)**:
    *   **这是你要求的重点**。我们不直接用 $E_k$ 作为结果。
    *   定义一组 MLP (可以所有 Code 共享一个大 MLP，也可以分组)。
    *   **输入**: $H_{motif}^{2D}$ (2D 上下文) + $E_k$ (选中的原型)。
    *   **输出**: **形变向量 $\Delta z$**。
    *   **最终 3D 预测**: $\hat{z}_{motif}^{3D} = E_k + \Delta z$。
    *   *物理意义*：$E_k$ 说是“苯环”，$\Delta z$ 说是“因为连了硝基，所以平面稍微扭曲了一点”。

---

## Step 3: 预训练 (Pre-training)

**数据**: PCQM4Mv2 (SMILES + 3D Labels)。
**流程**:

1.  **2D 路径**: SMILES $\rightarrow$ GIN $\rightarrow$ Motif Pooling $\rightarrow$ $H_{motif}^{2D}$。
2.  **3D 路径**: 3D Coords $\rightarrow$ 计算真实局部几何特征 $z_{3D}^{GT}$。
3.  **VQ-MoE 前向**:
    *   Router 根据 $H_{motif}^{2D}$ 选出 Code $k$。
    *   MLP 根据上下文预测形变，得到 $\hat{z}_{motif}^{3D}$。
4.  **计算 Loss**:

$$ \mathcal{L}_{Total} = \mathcal{L}_{Recon} + \mathcal{L}_{VQ} + \mathcal{L}_{Commit} $$

*   **$\mathcal{L}_{Recon}$ (MLP 拟合能力)**:
    *   $\text{MSE}(\hat{z}_{motif}^{3D}, z_{3D}^{GT})$。
    *   *解释*：强迫 MLP Expert 准确预测出真实的 3D 几何特征（键长、角度等）。
*   **$\mathcal{L}_{VQ}$ (Codebook 聚类)**:
    *   $\| \text{sg}[z_{3D}^{GT}] - E_k \|_2^2$。
    *   *解释*：让 Codebook 的向量 $E_k$ 真正成为一类几何形状的中心（聚类中心）。
*   **$\mathcal{L}_{Commit}$ (2D 映射一致性)**:
    *   $\| H_{motif}^{2D} - \text{sg}[E_k] \|_2^2$。
    *   *解释*：让 2D Embedding 不要离选中的 Code 太远。

---

## Step 4: 微调 (Fine-tuning)

**数据**: MoleculeNet (BBBP, ESOL 等)。
**状态**: 丢弃 3D 数据处理部分，**冻结 Codebook**，但**微调 Expert MLP**。

1.  **输入**: 仅 SMILES。
2.  **Motif 增强**:
    *   GIN 提取 $H_{motif}^{2D}$。
    *   Router 自动“回忆”出它是哪类几何原型 (Code $k$)。
    *   Expert MLP 根据当前的 2D 上下文，实时计算出 3D 形变 $\Delta z$。
    *   得到增强特征：$H_{enhanced} = \text{Concat}(H_{motif}^{2D}, E_k + \Delta z)$。
3.  **全局聚合 (Readout)**:
    *   现在我们有一堆 Motif 的特征，需要聚合成整分子特征。
    *   使用 **Attention Pooling** 或 **Sum Pooling** 将所有 Motif 的 $H_{enhanced}$ 聚合成一个向量 $H_{mol}$。
4.  **预测**:
    *   $H_{mol} \rightarrow$ Prediction Head $\rightarrow$ 属性值。

**Loss**:
*   $\text{MSE}(\text{Pred}, \text{Label})$ 或 $\text{BCE}(\text{Pred}, \text{Label})$。
