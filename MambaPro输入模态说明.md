# MambaPro输入模态说明

## 🎯 核心结论

**每次输入都是三种模态一起的！**

MambaPro采用**多模态并行输入**的设计，每次前向传播都会同时处理RGB、NI（近红外）、TI（热红外）三种模态。即使某些数据集只有两种模态，也会创建虚拟的第三种模态以保持接口一致性。

---

## 📊 输入格式详解

### 1. **模型输入格式**

```python
# 模型前向传播的输入格式
x = {
    'RGB': RGB_tensor,    # [B, 3, 256, 128] - 可见光图像
    'NI':  NI_tensor,     # [B, 3, 256, 128] - 近红外图像
    'TI':  TI_tensor      # [B, 3, 256, 128] - 热红外图像
}
```

### 2. **数据加载流程**

```python
# data/datasets/make_dataloader.py
def train_collate_fn(batch):
    imgs, pids, camids, viewids, _ = zip(*batch)
    
    RGB_list = []
    NI_list = []
    TI_list = []
    
    # 从每个样本中提取三种模态
    for img in imgs:
        RGB_list.append(img[0])  # RGB模态
        NI_list.append(img[1])   # NI模态
        TI_list.append(img[2])   # TI模态
    
    # 堆叠成batch
    RGB = torch.stack(RGB_list, dim=0)  # [B, 3, 256, 128]
    NI = torch.stack(NI_list, dim=0)    # [B, 3, 256, 128]
    TI = torch.stack(TI_list, dim=0)    # [B, 3, 256, 128]
    
    # 返回字典格式
    imgs = {'RGB': RGB, "NI": NI, "TI": TI}
    return imgs, pids, camids, viewids
```

---

## 🔍 不同数据集的模态处理

### **1. RGBNT201（三模态数据集）**

```python
# data/datasets/RGBNT201.py
# 每个样本有三个独立的图像路径
img_list = [rgb_path, nir_path, tir_path]  # 三个不同的路径

# 数据加载
def read_image(img_list):
    img3 = []
    for i in img_list:  # 遍历三个路径
        img = Image.open(i).convert('RGB')
        img3.append(img)  # 直接加入列表
    return img3  # [RGB图像, NIR图像, TIR图像]
```

**特点**：
- ✅ 三种模态都是真实的、独立的图像
- ✅ RGB、NIR、TIR分别来自不同的图像文件

### **2. RGBNT100（双模态数据集）**

```python
# data/datasets/RGBNT100.py
# 每个样本只有一个图像路径（包含RGB和IR）
img_list = [img_path, img_path, img_path]  # 三个相同的路径（为了兼容）

# 数据加载
def read_image(img_list):
    if len(set(img_list)) == 1:  # 所有路径相同
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        
        # 从单张图像中提取RGB和IR
        RGB = img.crop((0, 0, width//2, height))      # 左半部分：RGB
        IR = img.crop((width//2, 0, width, height))   # 右半部分：IR
        
        # 🔥 关键：创建虚拟的TI（使用IR图像）
        img3 = [RGB, IR, IR]  # RGB, IR, 虚拟TI
    return img3
```

**特点**：
- ✅ RGB和IR是真实的、从单张图像中提取的
- ⚠️ TI是虚拟的（使用IR的副本），用于保持接口一致性

### **3. MSVR310（三模态数据集）**

```python
# data/datasets/bases.py
# 单张图像包含三种模态（水平拼接）
def read_image(img_path):
    img = Image.open(img_path).convert('RGB')
    
    # 将图像切割成三部分
    RGB = img.crop((0, 0, 256, 128))      # [0:256, 0:128] - RGB
    NI = img.crop((256, 0, 512, 128))     # [256:512, 0:128] - NI
    TI = img.crop((512, 0, 768, 128))     # [512:768, 0:128] - TI
    
    img3 = [RGB, NI, TI]
    return img3
```

**特点**：
- ✅ 三种模态都是从单张拼接图像中切割出来的
- ✅ 图像格式：768×128（RGB:256 + NI:256 + TI:256）

---

## 🚀 模型前向传播流程

### **训练时**

```python
# modeling/make_model.py
def forward(self, x, label=None, cam_label=None, view_label=None):
    if self.training:
        # 🔥 同时提取三种模态
        RGB = x['RGB']  # [B, 3, 256, 128]
        NI = x['NI']    # [B, 3, 256, 128]
        TI = x['TI']    # [B, 3, 256, 128]
        
        # 🔥 并行处理三种模态
        RGB_tokens, RGB_score, RGB_global = self.BACKBONE(
            RGB, cam_label=cam_label, view_label=view_label, modality='rgb'
        )
        NI_tokens, NI_score, NI_global = self.BACKBONE(
            NI, cam_label=cam_label, view_label=view_label, modality='nir'
        )
        TI_tokens, TI_score, TI_global = self.BACKBONE(
            TI, cam_label=cam_label, view_label=view_label, modality='tir'
        )
        
        # 🔥 检测数据集类型（双模态 vs 三模态）
        is_dual_modal = torch.allclose(NI_global, TI_global, atol=1e-6)
        
        if is_dual_modal:
            # RGBNT100：只使用RGB和IR
            ori = torch.cat([RGB_global, NI_global], dim=-1)
            fuse = self.AAM(RGB_cash, NI_cash, None)  # TI传入None
        else:
            # RGBNT201：使用RGB、NI、TI
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            fuse = self.AAM(RGB_cash, NI_cash, TI_cash)
```

### **测试时**

```python
def forward(self, x, label=None, cam_label=None, view_label=None):
    else:  # 测试模式
        RGB = x['RGB']
        NI = x['NI']
        TI = x['TI']
        
        # 同样并行处理三种模态
        RGB_cash, RGB_global = self.BACKBONE(RGB, ...)
        NI_cash, NI_global = self.BACKBONE(NI, ...)
        TI_cash, TI_global = self.BACKBONE(TI, ...)
        
        # 融合输出
        fuse = self.AAM(RGB_cash, NI_cash, TI_cash)
        return fuse
```

---

## 📈 多尺度滑动窗口在三种模态中的应用

### **每个模态都独立应用多尺度滑动窗口**

```python
# 对于RGB模态
RGB_tokens = self.BACKBONE(RGB, modality='rgb')
# → CLIP提取patch tokens [B, 129, 512]
# → 多尺度滑动窗口处理
#   - 4×4窗口：局部细节特征
#   - 8×8窗口：中等结构特征
#   - 16×16窗口：全局上下文特征
# → 多尺度融合特征 [B, 512]

# 对于NI模态（同样处理）
NI_tokens = self.BACKBONE(NI, modality='nir')
# → 同样的多尺度滑动窗口处理

# 对于TI模态（同样处理）
TI_tokens = self.BACKBONE(TI, modality='tir')
# → 同样的多尺度滑动窗口处理
```

### **关键点**

1. **每个模态独立处理**：RGB、NI、TI分别通过相同的backbone和多尺度滑动窗口
2. **共享backbone**：三种模态共享同一个CLIP视觉编码器（通过`modality`参数区分）
3. **多尺度特征独立提取**：每个模态都有自己的多尺度特征表示
4. **最终融合**：三种模态的多尺度特征通过AAM（Mamba聚合）融合

---

## 🎯 总结

### **输入特点**

1. ✅ **总是三种模态**：每次输入都包含RGB、NI、TI三种模态
2. ✅ **字典格式**：输入格式为`{'RGB': tensor, 'NI': tensor, 'TI': tensor}`
3. ✅ **并行处理**：三种模态同时通过backbone提取特征
4. ✅ **多尺度应用**：每个模态都独立应用多尺度滑动窗口

### **数据集差异**

| 数据集 | RGB | NI | TI | 说明 |
|--------|-----|----|----|------|
| **RGBNT201** | ✅ 真实 | ✅ 真实 | ✅ 真实 | 三模态独立图像 |
| **RGBNT100** | ✅ 真实 | ✅ 真实 | ⚠️ 虚拟（IR副本） | 双模态，TI为占位符 |
| **MSVR310** | ✅ 真实 | ✅ 真实 | ✅ 真实 | 三模态拼接图像 |

### **设计优势**

1. **接口统一**：所有数据集使用相同的输入格式，便于代码复用
2. **灵活兼容**：通过检测特征相似性自动识别双模态/三模态数据集
3. **并行处理**：三种模态同时处理，提高计算效率
4. **多尺度增强**：每个模态都受益于多尺度滑动窗口的特征提取

---

## 📚 代码位置

- **数据加载**：`data/datasets/make_dataloader.py` - `train_collate_fn()`
- **图像读取**：`data/datasets/bases.py` - `read_image()`
- **模型前向**：`modeling/make_model.py` - `MambaPro.forward()`
- **训练循环**：`engine/processor.py` - `do_train()`




