#!/usr/bin/env python
# run.py
import os
import sys
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from preprocess.py11_file_utils import map_brain_to_days_and_vice_versa

brain_region_list = ['left_ALM', 'left_BLA', 'left_ECT', 'left_Medulla', 'left_Midbrain', 'left_Striatum', 'left_Thalamus'] +\
                    ['right_ALM', 'right_BLA', 'right_ECT', 'right_Medulla', 'right_Midbrain', 'right_Striatum', 'right_Thalamus']

# ========= 可改参数 =========
BRAIN_REGION   = "left_Medulla"
TASK_NAME      = "constant_init_fr_no_ls"   # 仅作提示，未实际用到
N_limit        = 4                          # 最大并发进程数

PY_SCRIPT      = "./analyse/py0_single.py"
LOG_DIR        = "/work1/yuhan/neuralflow/logs"
PY_DIR         = "/work1/yuhan/neuralflow"
# ============================

# 激活 conda 环境（如果已在 base 里运行可删掉）
CONDA_ENV = "nf"

def init_conda():
    """让当前 Python 继承 conda 环境变量"""
    # 找到 conda.sh 路径
    conda_base = subprocess.check_output("conda info --base", shell=True, text=True).strip()
    conda_sh   = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
    # 把 conda 环境变量导入到 os.environ
    cmd = f"source {conda_sh} && conda activate {CONDA_ENV} && python -c 'import os,json;print(json.dumps(dict(os.environ)))'"
    env_dict = subprocess.check_output(cmd, shell=True, executable="/bin/bash", text=True).strip()
    os.environ.update(eval(env_dict))

def run_one(alpha: float):
    """跑单个 alpha 的任务"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = f"{LOG_DIR}/{timestamp}_py0_single_{BRAIN_REGION}_{DATE}_no_ls_alpha_{alpha}.log"

    cmd = [
        sys.executable, PY_SCRIPT,
        "--brain_region", BRAIN_REGION,
        "--date", DATE,
        "--init_fr", "constant",
        "--task_name", f"cfr_no_ls_alpha_{alpha}",
        "--alpha", str(alpha)
    ]

    with open(log_file, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PY_DIR)
        return p.wait()   # 等待子进程结束

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    init_conda()

    mapping_brain_to_days, mapping_days_to_brain = map_brain_to_days_and_vice_versa(brain_region_list)         
    dates = mapping_brain_to_days[BRAIN_REGION]
    alphas = [0.05, 0.01, 0.005, 0.001, 0.0005]

    # 把 (date, alpha) 两两拼成任务
    tasks = [(d, a) for d in dates for a in alphas]

    with ThreadPoolExecutor(max_workers=N_limit) as pool:
        futures = [pool.submit(run_one, date, alpha) for date, alpha in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print("任务异常:", exc, file=sys.stderr)

    print("全部任务已结束。")

if __name__ == "__main__":
    main()