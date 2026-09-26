# 问题3服务器运行说明

入口为 `question3/q3_run.py`；便捷脚本为同目录的 `run_q3_server.sh`。不依赖仓库根目录的旧启动脚本或 `server.env.sh`，不调用带实验版本号的源文件。

## 环境与文件

沿用服务器已有CUDA PyTorch环境和BERT缓存：

```bash
conda activate MathE
cd /root/user/cs_tcci_yeguoao/MathE
python -m pip install -r question3/requirements.txt

export MATHE_DEVICE=cuda:0
export Q3_DATA_ROOT=/root/user/cs_tcci_yeguoao/MathE/E题数据
export Q3_BERT=/root/user/cs_tcci_yeguoao/MathE/work/pretrained_models/bert
```

`Q3_DATA_ROOT`须直接包含 `附件2-数据集特征文件/aligned_50.pkl` 和附件4目录。若实际数据多嵌套了一层，改为实际路径。未设置时，脚本自动检查根目录下 `E题数据/` 和 `E题数据/E题数据/`。BERT指纹必须与已有部署记录匹配。

## 检查与新训练

仅整理代码不需要重新训练或重算已保存结果。开展新的完整训练时，使用新的运行目录：

```bash
export Q3_RUN=q3_final_01
bash -n question3/run_q3_server.sh
bash question3/run_q3_server.sh test
bash question3/run_q3_server.sh check
bash question3/run_q3_server.sh train
cat outputs/question3/q3_final_01/目标检查.json
```

默认比较四个候选，首种子胜出结构再训练另外两个种子，并比较单模型及组合的部署规则；最后不一定选择集成。已归档结果实际选择一个 `token_interaction_hierarchical` 组件。默认最多20轮、patience 8、batch 16、种子2026/2027/2028。历史对照组键 `v3_control` 保留以兼容记录，模型与训练参数未因文件整理而改变。

同一运行目录包含代码哈希校验。源文件重命名后，不得直接在旧目录继续 `train`；恢复被中断的旧训练应使用原代码和原配置，整理后代码重训应创建新目录。

## 已完成训练的解释与审计

服务器保留原始运行目录、组件权重与配置时，可只重做解释、报告或审计：

```bash
export Q3_OUT=/root/user/cs_tcci_yeguoao/MathE/outputs/question3/q3_v5_01
bash question3/run_q3_server.sh explain
bash question3/run_q3_server.sh report
bash question3/run_q3_server.sh audit
unset Q3_OUT
```

旧目录名仅标识已有实验，不是代码入口。`Q3_OUT`优先于`Q3_RUN`。解释核对组件SHA-256、BERT指纹和官方数据哈希，解释目标包含实际部署类别偏置。

GitHub五类目录是论文归档格式；`explain/report/audit`需要原始运行格式，即同一目录下有 `模型参数/`、`配置与审计/`、`验证集评价/` 等。不要把归档目录的 `01_建模原理与训练方案/` 当作完整运行目录。

附件4未核验的音视频证据仍只定位到特征行。`mapping`/`media`工具不会把未核验的对应关系变成精确秒级证据。

## 打包源码

```bash
python question3/package_q3_server.py
```

生成 `outputs/question3/问题3_服务器代码包.zip`。源码均位于 `question3/`，含本目录启动脚本和依赖说明，不包含数据、BERT权重或训练结果。解压后按上述命令运行。
