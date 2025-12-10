# PowerShell 乱码问题解决方案

## 问题原因
文件 `modeling\fusion_part\multi_scale_moe.py` 包含中文注释，使用 UTF-8 编码，但 PowerShell 默认可能使用其他编码读取，导致乱码。

## 解决方案

### 方案 1：指定 UTF-8 编码读取文件（推荐）

```powershell
Get-Content "modeling\fusion_part\multi_scale_moe.py" -Encoding UTF8 | Select-String -Pattern "enhanced_multi_scale_features.*gate_fusion" -Context 5,5 | Select-Object -First 1
```

### 方案 2：设置 PowerShell 输出编码为 UTF-8

在执行命令前，先设置输出编码：

```powershell
# 设置控制台输出编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 然后执行原命令
Get-Content "modeling\fusion_part\multi_scale_moe.py" | Select-String -Pattern "enhanced_multi_scale_features.*gate_fusion" -Context 5,5 | Select-Object -First 1
```

### 方案 3：使用 chcp 命令设置代码页

```powershell
# 设置代码页为 UTF-8 (65001)
chcp 65001

# 然后执行原命令
Get-Content "modeling\fusion_part\multi_scale_moe.py" | Select-String -Pattern "enhanced_multi_scale_features.*gate_fusion" -Context 5,5 | Select-Object -First 1
```

### 方案 4：使用 findstr（Windows 原生工具）

```cmd
findstr /N /C:"enhanced_multi_scale_features" "modeling\fusion_part\multi_scale_moe.py"
```

### 方案 5：使用 Python 脚本（跨平台）

创建一个 Python 脚本来搜索：

```python
import re

with open("modeling/fusion_part/multi_scale_moe.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    
pattern = re.compile(r"enhanced_multi_scale_features.*gate_fusion")
for i, line in enumerate(lines, 1):
    if pattern.search(line):
        start = max(0, i - 6)
        end = min(len(lines), i + 5)
        print(f"Line {i}:")
        for j in range(start, end):
            print(f"{j+1:4d}: {lines[j]}", end="")
        break
```

## 推荐使用方案 1

最简单直接，只需在 `Get-Content` 后添加 `-Encoding UTF8` 参数即可。

## 预期输出

正确执行后，应该看到类似这样的输出：

```
modeling\fusion_part\multi_scale_moe.py:694:        enhanced_multi_scale_features, gate_weights = self.gate_fusion(multi_scale_features)
```

