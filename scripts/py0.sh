#!/bin/bash
# 初始化 Conda
eval "$(conda shell.bash hook)"
conda activate nf

BRAIN_REGION="left_Medulla"
DATE="57"

# 定义日志目录和Python脚本路径
LOG_DIR="/work1/yuhan/neuralflow/logs"
PY_DIR="/work1/yuhan/neuralflow/"
PY_SCRIPT="./analyse/py0_try.py"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 设置 Python 路径
export PYTHONPATH="/work1/yuhan/neuralflow:$PYTHONPATH"

# nohup后台运行，并将输出写入日志文件
cd $PY_DIR
# nohup python "$PY_SCRIPT" --brain_region "$BRAIN_REGION" --date "$DATE" > "$LOG_DIR/py0_try_${BRAIN_REGION}_${DATE}.log" 2>&1 &
python "$PY_SCRIPT" --brain_region "$BRAIN_REGION" --date "$DATE" > "$LOG_DIR/py0_try_${BRAIN_REGION}_${DATE}.log"

echo "已后台启动 $PY_SCRIPT, 参数: brain_region=${BRAIN_REGION}, date=${DATE}"