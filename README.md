# 本科物理的数学基础 (Mathematical Foundations for Undergraduate Physics)

> 物理为血肉，数学为骨架 —— 在物理情景中理解数学，从数学推导中洞察物理。

本仓库是《本科物理的数学基础》一书的 LaTeX 源码。全书共五章，从物理问题出发，
系统地构建本科物理所需的数学工具：

| 章 | 内容 | 物理连接 |
|----|------|----------|
| 第 1 章 | 坐标系、矢量分析与张量初步 | 电磁学、力学、量子力学预备 |
| 第 2 章 | 复变函数论 | 相量法、留数定理算实积分、保角映射、Kramers-Kronig 关系 |
| 第 3 章 | 积分变换：傅里叶分析与拉普拉斯变换 | 信号处理、电路、不确定性原理 |
| 第 4 章 | 偏微分方程与特殊函数 | 弦振动、热传导、圆膜、氢原子、谐振子 |
| 第 5 章 | 数值方法：有限差分与 SciPy 计算 | PDE 的计算机求解、FFT、矩阵本征值 |

## 编译方法

需要 TeX Live（或 MiKTeX）并启用 XeLaTeX 与 ctex 宏包：

```bash
latexmk -xelatex main.tex
# 或手工编译两次：
xelatex main.tex
xelatex main.tex
```

编译产物为 `main.pdf`。

## 目录结构

```
main.tex                                  # 主文件（页面设置、宏包、章节包含）
chapters/chapterN_*.tex                   # 各章正文
figures/                                  # 插图（PNG）与生成脚本
figures/generate_all_figures.py           # 全部插图的生成脚本（Python + matplotlib）
```

## 环境要求

- XeLaTeX + ctexbook 文档类（中文排版）
- 常见宏包：amsmath、physics、bm、tcolorbox、booktabs、listings 等（TeX Live 完整安装即可）
- 插图脚本：Python 3 + NumPy + SciPy + matplotlib

## 许可

仅供学习交流使用。
