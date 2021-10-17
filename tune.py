#

# a simple script to tune things

import multiprocessing
from multiprocessing import Pool, Lock, Manager
import subprocess
import os
import sys
import time
import numpy as np
np.random.seed(12345)

# --
# global lock!
_global_lock = Lock()
manager = multiprocessing.Manager()
Global = manager.Namespace()
Global.idx = 0
Global.gpu_available = ''  # note: cannot be a complex object, which will not be sync
_global_log = "_stdout.log"
# --

# def run_cmd(cmd: str):
#     try:
#         tmp_out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
#         n = 0
#         output = str(tmp_out.decode())  # byte->str
#     except subprocess.CalledProcessError as grepexc:
#         n = grepexc.returncode
#         output = grepexc.output
#     return output

def run_cmd(cmd: str):
    print(f"Run {cmd}")
    # return os.system(cmd)
    ret = subprocess.run(cmd, shell=True)
    return ret

def run_one(arg_str: str):
    # --
    gpu_id = None
    while True:  # not getting resource?
        with _global_lock:
            print(f"{arg_str} {Global.gpu_available}")
            for ii, vv in enumerate(Global.gpu_available):
                if vv == '1':
                    gpu_id = ii
                    break
            if gpu_id is not None:
                _rs = list(Global.gpu_available)
                _rs[gpu_id] = '0'
                Global.gpu_available = ''.join(_rs)  # take it!!
                cur_idx = Global.idx
                Global.idx = Global.idx + 1
                print(f"Claim {gpu_id} with {cur_idx}, currently {Global.gpu_available}")
                break
            else:  # otherwise wait for some time
                print("Resource not found, wait ...")
                time.sleep(10)
    print(f"Start task {cur_idx}: {arg_str}")
    # --
    _log_suffix = '_'.join(''.join(arg_str.split("--")).split())
    # run_cmd(f"CUDA_VISIBLE_DEVICES={gpu_id} EXTRA_ARGS='{arg_str} --qa_save_name zmodel{cur_idx}' bash ../train_qa.sh 2>&1 | tee _log{cur_idx}.{_log_suffix}")
    _train_cmd = "" if "NOTRAIN" in arg_str \
        else f"CUDA_VISIBLE_DEVICES={gpu_id} EXTRA_ARGS='{arg_str} --qa_save_name zmodel{cur_idx}' bash ../train_qa.sh;"
    run_cmd(f"""{{
        {_train_cmd}
        for tt in 0. 1. 2. 3. 4.; do
            echo "Decode pthr=${{tt}}"
            CUDA_VISIBLE_DEVICES={gpu_id} python3 ../qa_main.py --mode squad --input_path ../data/dev-v2.0.json --output_path '' --model zmodel{cur_idx}.best --model_kwargs "{{'qa_label_pthr':${{tt}}}}"
        done
    }} 2>&1 | tee _log{cur_idx}.{_log_suffix}
    """)
    # --
    with _global_lock:
        _rs = list(Global.gpu_available)
        _rs[gpu_id] = '1'
        Global.gpu_available = ''.join(_rs)  # put it back!
        print(f"End task {cur_idx}: {arg_str}")
    # --

def run_them(ranges: list, gpu_ids: list, shuffle=False):
    # --
    # put resources
    _rs = ['0'] * (max(gpu_ids) + 1)
    for ii in gpu_ids:
        _rs[ii] = '1'
    Global.gpu_available = ''.join(_rs)
    # --
    # first expand get all ranges
    all_args = [""]
    for one_ranges in ranges:
        new_all_args = []
        for a in all_args:
            for a2 in one_ranges:
                new_all_args.append(a+" "+a2)
        all_args = new_all_args
    # shuffle them all
    print(f"All tasks = {len(all_args)}")
    if shuffle:
        np.random.shuffle(all_args)
    # run them
    with Pool(len(gpu_ids)) as p:
        p.map(run_one, all_args)
    # --

# --
def main():
    # --
    tune_ranges1013 = [
        [f"--learning_rate {z}" for z in [1e-5, 3e-5, 5e-5]],
        [f"--qa_label_negratio {z}" for z in [5, 10]],
    ]
    # 0: best=4.: 73.014/77.151;; 1: best=3.: 74.159/78.062
    # 2: best=3.: 75.760/79.639;; 3: best=3.: 76.299/79.989 [best]
    # 4: best=3.: 75.086/78.808;; 5: best=3.: 75.254/79.302
    # --
    tune_ranges1015 = [
        [f"--qa_label_negratio {z}" for z in [10, 20]],
        [f"--qa_label_ls {z}" for z in [0., 0.05]],
    ]
    dec_ranges1015 = tune_ranges1015 + [["NOTRAIN"]]
    # 0: best=3.: 76.299/79.989;; 1: best=2.: 76.080/79.895
    # 2: best=2.: 76.703/80.400;; 3: best=1.: 77.040/80.896 [best]
    # --
    tr = dec_ranges1015
    run_them(tr, [1,3])

if __name__ == '__main__':
    main()

"""
# examples runs with "qa_main.py"
# decode with ptr-models
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model try0/zmodel.best
# -> OrderedDict([('exact', 77.50357955024005), ('f1', 80.42150998666928), ('total', 11873), ('HasAns_exact', 72.62145748987854), ('HasAns_f1', 78.46568624691706), ('HasAns_total', 5928), ('NoAns_exact', 82.3717409587889), ('NoAns_f1', 82.3717409587889), ('NoAns_total', 5945)])
# decode with label-models
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model try1/zmodel.best --model_kwargs "{'qa_label_pthr':3.}"
# -> OrderedDict([('exact', 75.76012802156153), ('f1', 79.63949665083895), ('total', 11873), ('HasAns_exact', 72.80701754385964), ('HasAns_f1', 80.57687984740397), ('HasAns_total', 5928), ('NoAns_exact', 78.70479394449117), ('NoAns_f1', 78.70479394449117), ('NoAns_total', 5945)])
# --
# search for the tuned models
for ii in {0..5}; do
for tt in 0. 1. 2. 3. 4.; do
echo "RUN model${ii} pthr=${tt}"
python3 qa_main.py --mode squad --input_path data/dev-v2.0.json --output_path '' --model tune1013/zmodel${ii}.best --model_kwargs "{'qa_label_pthr':${tt}}"
done
done |& tee tune1013/_log_dec
cat tune1013/_log_dec | grep -E "RUN|Eval"
# -> the best is: rate3e-5,neg10 -> pthr=3 -> 76.299/79.989
# ======
"""
