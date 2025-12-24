import time
import random
import os
import math
from multiprocessing import Pool
import subprocess
import threading
from queue import Queue
import sys
from datetime import datetime, timedelta




def split_list(lst, x):
    size = len(lst)
    chunk_size = size // x
    remainder = size % x

    result = []
    start = 0
    for i in range(x):
        end = start + chunk_size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result


def output_reader(pipe, queue):
    try:
        for line in iter(pipe.readline, b''):
            queue.put(line.decode())
    finally:
        pipe.close()


def run_command(command):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               shell=True,
                               bufsize=1)

    stdout_queue = Queue()
    stderr_queue = Queue()

    stdout_thread = threading.Thread(target=output_reader,
                                     args=(process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=output_reader,
                                     args=(process.stderr, stderr_queue))

    stdout_thread.start()
    stderr_thread.start()

    while True:
        if process.poll() is not None and stdout_queue.empty() and stderr_queue.empty():
            break

        while not stdout_queue.empty():
            print(stdout_queue.get(), end='')

        while not stderr_queue.empty():
            print(f"ERROR: {stderr_queue.get()}", end='', file=sys.stderr)

    stdout_thread.join()
    stderr_thread.join()

    return process.returncode


def parallel_execute(script_list):
    threads = []
    for command in script_list:
        thread = threading.Thread(target=run_command, args=(command,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()



# training code
## Ovtrack fixed pair training code
def train_network(train_work_dir, train_cfg_file, options=None):
    train_port = number = random.randint(20001, 29999)
    command = [
        "CUDA_VISIBLE_DEVICES=0,1,2,3",
        "tools/dist_train.sh",
        f"{train_cfg_file}",
        "4",
        # "23349",
        f"{train_port}",
        "--work-dir",
        f"{train_work_dir}"
    ]
    if options is not None:
        command.append("--cfg-options")
        command.append(options)


    print(f"Using script {' '.join(command)}")
    # exec training code
    subprocess.run(" ".join(command), shell=True)

    return command



# inference code
def test_network_with_checkpoint_dir(test_cfg_file, test_epoch_file, test_epoch_dir=None, split=1, options=None):
    test_port = random.randint(30001, 49999)
    if test_epoch_dir is not None:
        checkpoint_files = sorted(
            [os.path.join(test_epoch_dir, file) for file in os.listdir(test_epoch_dir) if 'epoch' in file],
            key=lambda f: int(f.split('_')[-1].split('.')[0]))
    script_list = []
    if split == 1:
        command = [
            "CUDA_VISIBLE_DEVICES=0,1,2,3",
            "tools/dist_test.sh",
            f"{test_cfg_file}",
            f"{test_epoch_file}",

            "4",
            # "34689",
            f"{test_port}",

            "--eval track",

            "--eval-options",

            "resfile_path=results/debug3",


            "--checkpoint-dir",
            f"{test_epoch_dir}",

            # "--cfg-options",
            # "model.tracker.match_score_thr=0.35"

        ]
        if options is not None:
            command.append("--cfg-options")
            command.append(options)
        script_list.append(' '.join(command))
    else:
        assert split > 1, f"Split length is {split}"
        split_indices = split_list(list(range(len(checkpoint_files))), split)
        for inner_list in split_indices:
            test_port = random.randint(30001, 49999)
            l_ = min(inner_list)
            r_ = max(inner_list)
            command = [
                "CUDA_VISIBLE_DEVICES=0,1,2,3",
                "tools/dist_test.sh",
                f"{test_cfg_file}",
                f"{test_epoch_file}",

                "4",
                # "34689",
                f"{test_port}",

                "--eval track",

                "--eval-options",

                "resfile_path=results/debug3",


                "--checkpoint-dir",
                f"{test_epoch_dir}",

                f"--checkpoint-l-index {l_} --checkpoint-r-index {r_}",

                # "--cfg-options",
                # "model.tracker.match_score_thr=0.35"
            ]
            if options is not None:
                command.append("--cfg-options")
                command.append(options)
            script_list.append(' '.join(command))

    print(script_list)
    parallel_execute(script_list)
    # for command in script_list:
    #     # print(command)
    #     print(f"Using script {command}")
    #     # print(f"Using script {' '.join(command)}")
    #     # exec inference code
    #     subprocess.run(" ".join(command), shell=True)
    return script_list





if __name__ == '__main__':
    # p1
    train_cfg_file = 'configs/uncertainty-ovtrack-teta/ovtrack_r50_ctao_train.py'

    train_work_dir = 'work_dirs/VOVTrack_after/c_tao_training_tidy'
    train_options = "model.roi_head.feature_fusion_head.max_fusion_ratio=2.0 data.train.dataset.dataset.extra_sample_ratio=8 model.roi_head.cyc_loss_start_iteration=1000 total_epochs=20 data.train.dataset.dataset.ref_img_sampler.scope=30 model.roi_head.use_cyc_loss=False"
    train_start_time = time.time()
    train_cmd = train_network(train_work_dir, train_cfg_file, options=train_options)
    train_end_time = time.time()


    
    test_cfg_file = train_cfg_file
    test_epoch_dir = train_work_dir
    test_epoch_file = os.path.join(test_epoch_dir, 'epoch_1.pth')

    test_options =  "model.tracker.match_score_thr=0.37 model.test_cfg.rcnn.max_per_img=80 model.roi_head.feature_fusion_head.max_fusion_ratio=2.0 model.tracker.confused_features=True model.roi_head.only_validation_categories=True model.tracker.memo_frames=50 model.tracker.momentum_embed=0.4"
    split_num = 2
    test_start_time = time.time()
    test_cmd_list = test_network_with_checkpoint_dir(test_cfg_file, test_epoch_file, test_epoch_dir, split_num, options=test_options)
    test_end_time = time.time()




    print(f"Train cmd : {' '.join(train_cmd)} in {train_end_time - train_start_time}, \n")
    for test_cmd in test_cmd_list:
        print(f"Test cmd : {str(test_cmd)}\n")
    print(f"Test time cost  {test_end_time - test_start_time}")




