#!/bin/bash
# 初始化 Conda
eval "$(conda shell.bash hook)"
conda activate nf

CHOICE="all_union"
TASK_NAME="all_union_reflecting_constant_init_fr_no_ls_selected_neurons"
NEURON_NUM=20
MAX_CONCURRENT=20

# 定义日志目录和Python脚本路径
LOG_DIR="/work1/yuhan/neuralflow/logs"
PY_DIR="/work1/yuhan/neuralflow/"
PY_SCRIPT="./analyse/py4_single_new_data_mini.py"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 设置 Python 路径
export PYTHONPATH="/work1/yuhan/neuralflow:$PYTHONPATH"


# nohup后台运行，并将输出写入日志文件
cd $PY_DIR
for BRAIN_REGION in "left Medulla" "right Medulla"; do
    for alpha in 0.05 0.01 0.005 0.001 0.0005; do
        nohup python "$PY_SCRIPT" --choice "$CHOICE" --brain_region "$BRAIN_REGION" --init_fr "constant" --task_name "${TASK_NAME}_${alpha}_neurons${NEURON_NUM}" --alpha "$alpha" --boundary "ref" --select_neurons --num_select_neurons "$NEURON_NUM" > "$LOG_DIR/$(date +'%Y%m%d_%H%M%S')_py4_single_${BRAIN_REGION// /_}_selected_neurons_${NEURON_NUM}_alpha_${alpha}.log" 2>&1 &
    done 
done