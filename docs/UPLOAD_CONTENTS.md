# 投稿代码仓库的文件安排

## 七个主要入口

| 内容 | 文件 | 用途 |
| --- | --- | --- |
| 精确算法 | `algorithms/exact.py` | 默认 DB、平均收益；支持 PC、IM 和累计收益，直接计算稀突变极限 |
| 近似算法 | `algorithms/approximate.py` | 论文主模型 DB、平均收益；默认每个节点 1,000 次短轨迹采样 |
| 仿真 | `algorithms/simulate.py` | 可读的 Python 仿真入口，支持三种更新规则、两种收益 |
| 网络生成 | `algorithms/generate_network.py` | 统一生成各网络家族，保存边列表和参数 |
| Fig. 2 | `figures/reproduce_fig2.py` | 独立读取 Fig. 2 的源数据并绘图 |
| Fig. 3 | `figures/reproduce_fig3.py` | 独立读取 Fig. 3 的源数据并绘图 |
| Fig. 4 | `figures/reproduce_fig4.py` | 独立读取 Fig. 4 的源数据并绘图 |

## 需要一起上传的配套文件

- `code/` 与 `algorithms/*_support/`：入口实际调用的数值求解与生产仿真代码。
- `figures/` 中的源数据及绘图辅助代码：让读者安装依赖后直接重绘。
- `examples/`：一个固定的小网络与快速运行示例。
- `tests/`：小图精确验证、模型约定检查和基本数值检查。
- `README.md`、依赖清单、`CITATION.cff`、`docs/`、GitHub Actions 配置。

图形重绘使用冻结的计算结果。重新生成这些结果需要运行相应算法及生产规模的
实验；两者的工作量不同，文档分别说明。完整经验网络下载和补充图流水线尚未
纳入此次整理范围。

工作目录、临时结果、旧版本图片及完整论文 PDF 不作为本次代码包内容。
