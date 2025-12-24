import argparse
import shutil
import os
import math
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
# from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataset
import datetime
import fcntl
import time
from ovtrack.datasets import off_eval_by_multi_thread


def parse_args():
    parser = argparse.ArgumentParser(description='ovtrack test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', type=str, nargs='+', help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--show_score_thr', default=0.3, type=float, help='output result file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--early-stop', action='store_true', help='Flag to determine if early stopping should be performed used to debug')
    parser.add_argument('--off-eval', action='store_true', help='Flag to determine if use offline evaluation')
    # used to search the best alpha_prob
    parser.add_argument('--l-alpha-prob', type=float, default=-1, help='[l_alpha_prob, r_alpha_prob] interval is 0.05')
    parser.add_argument('--r-alpha-prob', type=float, default=-1, help='[l_alpha_prob, r_alpha_prob] interval is 0.05')
    parser.add_argument('--alpha-prob-interval', type=float, default=0.05, help='[l_alpha_prob, r_alpha_prob] interval is 0.05')
    # used to search the best alpha_prob_thres
    parser.add_argument('--l-alpha-prob-thres', type=float, default=-1, help='[l_alpha_prob_thres, r_alpha_prob_thres] interval is 0.01')
    parser.add_argument('--r-alpha-prob-thres', type=float, default=-1, help='[l_alpha_prob_thres, r_alpha_prob_thres] interval is 0.01')
    parser.add_argument('--alpha-prob-thres-interval', type=float, default=0.01, help='[l_alpha_prob_thres, r_alpha_prob_thres] interval is 0.01')
    # used to search the best match score
    parser.add_argument('--l-match-score-thres', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')
    parser.add_argument('--r-match-score-thres', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')
    parser.add_argument('--match-score-thres-interval', type=float, default=0.01, help='[l_alpha_prob_thres, r_alpha_prob_thres] interval is 0.01')
    parser.add_argument('--l-max-bbox', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')
    parser.add_argument('--r-max-bbox', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')
    parser.add_argument('--max-bbox-interval', type=float, default=10, help='[l_alpha_prob_thres, r_alpha_prob_thres] interval is 0.01')
    parser.add_argument('--checkpoint-dir', default='', help='checkpoint directory for test')
    parser.add_argument('--checkpoint-l-index', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')
    parser.add_argument('--checkpoint-r-index', type=float, default=-1, help='[l_match_score_thres, r_match_score_thres] interval is 0.01')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def write_to_file(filepath, content):
    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        try:
            with open(filepath, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                f.write(f"{content}\n")
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return True

        except IOError as e:
            print(f"Process {os.getpid()} waiting... Attempt {attempt + 1}/{max_attempts}")
            time.sleep(1)  # 减少等待时间，使进程更频繁地尝试获取锁
            attempt += 1

    print(f"Process {os.getpid()} failed to write after maximum attempts")
    return False




def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.off_eval and not args.format_only:
        raise ValueError('--off-eval and --format_only must be both specified')


    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('USE_MMDET', False):
        from mmdet.apis import multi_gpu_test, single_gpu_test
        from mmdet.models import build_detector as build_model
        from mmdet.datasets import build_dataloader
    else:
        from ovtrack.apis import multi_gpu_test, single_gpu_test
        from ovtrack.models import build_model
        from ovtrack.datasets import build_dataloader

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        timeout = 72000
        cfg.dist_params['timeout'] = datetime.timedelta(seconds=timeout)
        # if 'timeout' in cfg.dist_params:
        #     cfg.dist_params['timeout'] = datetime.timedelta(seconds=cfg.dist_params['timeout'])
        init_dist(args.launcher, **cfg.dist_params)


    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=1, # fixed the value
        dist=distributed,
        shuffle=False)

    # grid search the match score
    # grid search the prob alpha thres
    if (args.l_match_score_thres != -1 and args.r_match_score_thres != -1) or ((args.l_max_bbox != -1 and args.r_max_bbox != -1)):
        epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
        match_score_range = []
        max_bbox_range = []

        # 处理match_score阈值对
        if args.l_match_score_thres != -1 and args.r_match_score_thres != -1:
            assert args.r_match_score_thres >= args.l_match_score_thres, "r match score must be greater than l match score"
            match_score_range = [args.l_match_score_thres + i * args.match_score_thres_interval for i in range(
                int((args.r_match_score_thres - args.l_match_score_thres) / args.match_score_thres_interval) + 1)]
        else:
            # 使用默认值
            match_score_range.append(cfg.model.tracker.match_score_thr)

        # 处理max_bbox对
        if args.l_max_bbox != -1 and args.r_max_bbox != -1:
            assert args.r_max_bbox >= args.l_max_bbox, "r max bbox must be greater than l max bbox"
            max_bbox_range = [args.l_max_bbox + i * args.max_bbox_interval for i in range(
                int((args.r_max_bbox - args.l_max_bbox) / args.max_bbox_interval) + 1)]
        else:
            # 使用默认值
            max_bbox_range.append(cfg.model.test_cfg.rcnn.max_per_img)


        # assert args.r_match_score_thres >= args.l_match_score_thres, "r alpha prob must bigger than l one"
        # for match_score_thr in [args.l_match_score_thres + i * args.match_score_thres_interval for i in range(math.ceil((args.r_match_score_thres - args.l_match_score_thres)/ args.match_score_thres_interval)+1)]:
        origianl_resfile_path = args.eval_options['resfile_path']
        eval_res_list = []
        new_head_list = []
        for match_score_thr in match_score_range:
            for max_bbox in max_bbox_range:
                # build the model and load checkpoint
                cfg.model.tracker.match_score_thr = match_score_thr
                cfg.model.test_cfg.rcnn.max_per_img = int(max_bbox)
                model = build_model(cfg.model, train_cfg=None, test_cfg=None)
                # model.roi_head.prob_thres = prob_thres
                print('reset the match schore thres as {} and max bbox as {}'.format(match_score_thr, max_bbox))
                model.roi_head.result_save_path = args.eval_options['resfile_path']
                # fp16_cfg = cfg.get('fp16', None)
                # if fp16_cfg is not None:
                #     wrap_fp16_model(model)
                checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

                if args.fuse_conv_bn:
                    model = fuse_conv_bn(model)


                model.CLASSES = dataset.CLASSES

                if not distributed:
                    model = MMDataParallel(model, device_ids=[0])
                    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                              args.show_score_thr, early_stop=args.early_stop)
                else:
                    model = MMDistributedDataParallel(
                        model.cuda(),
                        device_ids=[torch.cuda.current_device()],
                        broadcast_buffers=False)
                    outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                             args.gpu_collect)

                rank, _ = get_dist_info()
                if rank == 0:
                    if args.out:
                        print(f'\nwriting results to {args.out}')
                        mmcv.dump(outputs, args.out)
                    # update the result saved path
                    if args.off_eval:
                        args.eval_options['resfile_path'] = os.path.join(origianl_resfile_path, f'trackersearch_{round(match_score_thr,2)}_maxbbox_{int(max_bbox)}_epoch_{epoch}')
                        eval_res_list.append(args.eval_options['resfile_path'])
                    kwargs = {} if args.eval_options is None else args.eval_options
                    if args.format_only:
                        dataset.format_results(outputs, **kwargs)
                        new_head_list.append(f'epoch[{epoch}], match score thres[{round(match_score_thr, 2)} max bbox [{max_bbox}]]:')
                        continue
                    if args.eval:
                        eval_kwargs = cfg.get('evaluation', {}).copy()
                        # hard-code way to remove EvalHook args
                        for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                            eval_kwargs.pop(key, None)
                        eval_kwargs.update(dict(metric=args.eval, **kwargs))
                        eval_results = dataset.evaluate(outputs, **eval_kwargs)
                        print(eval_results)
                        combined_result = eval_results['combined_result']
                        base_result = eval_results['base_result']
                        novel_result = eval_results['novel_result']
                        epoch_line = f'epoch[{epoch}], match score thres[{round(match_score_thr, 2)} max bbox [{max_bbox}]]:'+ '\n' + combined_result+base_result+novel_result + '\n'
                    # with open(os.path.join(os.path.split(args.checkpoint)[0], 'tracker_match_score_search_eval_result.txt'), 'a') as f:
                    #     f.write(epoch_line)
                    write_to_file(os.path.join(os.path.split(args.checkpoint)[0], 'tracker_match_score_max_bbox_search_eval_result.txt'), epoch_line)
        if args.off_eval and rank == 0:
            val_dataset_path = dataset.ann_file
            off_eval_by_multi_thread(eval_res_list, os.path.split(args.checkpoint)[0], val_dataset_path, new_head_list, 'tracker_match_score_max_bbox_search_eval_result.txt')
        return



    # grid search the prob alpha thres
    if args.l_alpha_prob_thres != -1 and args.r_alpha_prob_thres != -1:
        epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
        assert args.r_alpha_prob_thres >= args.l_alpha_prob_thres, "r alpha prob must bigger than l one"
        for prob_thres in [args.l_alpha_prob_thres + i * args.alpha_prob_thres_interval for i in range(math.ceil((args.r_alpha_prob_thres - args.l_alpha_prob_thres)/ args.alpha_prob_thres_interval)+1)]:
            # build the model and load checkpoint
            model = build_model(cfg.model, train_cfg=None, test_cfg=None)
            model.roi_head.prob_thres = prob_thres
            print('reset the alpha prob thres as {}'.format(prob_thres))
            model.roi_head.result_save_path = args.eval_options['resfile_path']
            # fp16_cfg = cfg.get('fp16', None)
            # if fp16_cfg is not None:
            #     wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

            if args.fuse_conv_bn:
                model = fuse_conv_bn(model)


            model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                          args.show_score_thr, early_stop=args.early_stop)
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                         args.gpu_collect)

            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                if args.eval:
                    eval_kwargs = cfg.get('evaluation', {}).copy()
                    # hard-code way to remove EvalHook args
                    for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                        eval_kwargs.pop(key, None)
                    eval_kwargs.update(dict(metric=args.eval, **kwargs))
                    eval_results = dataset.evaluate(outputs, **eval_kwargs)
                    print(eval_results)
                    combined_result = eval_results['combined_result']
                    base_result = eval_results['base_result']
                    novel_result = eval_results['novel_result']
                    epoch_line = f'epoch[{epoch}], alpha prob thres[{prob_thres}]:'+ '\n' + combined_result+base_result+novel_result + '\n'
                with open(os.path.join(os.path.split(args.checkpoint)[0], 'prob_thres_search_eval_result.txt'), 'a') as f:
                    f.write(epoch_line)
        return


    # grid search the prob alpha
    if args.l_alpha_prob != -1 and args.r_alpha_prob != -1:
        epoch = int(args.checkpoint.split('_')[-1].split('.')[0])
        assert args.r_alpha_prob >= args.l_alpha_prob, "r alpha prob must bigger than l one"
        for alpha_prob in [args.l_alpha_prob + i * args.alpha_prob_interval for i in range(math.ceil((args.r_alpha_prob - args.l_alpha_prob)/ args.alpha_prob_interval)+1)]:
            # build the model and load checkpoint
            model = build_model(cfg.model, train_cfg=None, test_cfg=None)
            model.roi_head.prob_alpha = alpha_prob
            print('reset the alpha prob as {}'.format(alpha_prob))
            model.roi_head.result_save_path = args.eval_options['resfile_path']
            # fp16_cfg = cfg.get('fp16', None)
            # if fp16_cfg is not None:
            #     wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

            if args.fuse_conv_bn:
                model = fuse_conv_bn(model)


            model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                          args.show_score_thr, early_stop=args.early_stop)
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                         args.gpu_collect)

            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                if args.eval:
                    eval_kwargs = cfg.get('evaluation', {}).copy()
                    # hard-code way to remove EvalHook args
                    for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                        eval_kwargs.pop(key, None)
                    eval_kwargs.update(dict(metric=args.eval, **kwargs))
                    eval_results = dataset.evaluate(outputs, **eval_kwargs)
                    print(eval_results)
                    combined_result = eval_results['combined_result']
                    base_result = eval_results['base_result']
                    novel_result = eval_results['novel_result']
                    epoch_line = f'epoch[{epoch}], alpha prob[{alpha_prob}]:'+ '\n' + combined_result+base_result+novel_result + '\n'
                with open(os.path.join(os.path.split(args.checkpoint)[0], 'prob_search_eval_result.txt'), 'a') as f:
                    f.write(epoch_line)
        return

    if args.checkpoint_dir == '':
        # build the model and load checkpoint
        model = build_model(cfg.model, train_cfg=None, test_cfg=None)
        # fp16_cfg = cfg.get('fp16', None)
        # if fp16_cfg is not None:
        #     wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

        checkpoint_dir = os.path.split(args.checkpoint)[0]

        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)


        model.CLASSES = dataset.CLASSES

        model.roi_head.result_save_path = args.eval_options['resfile_path']
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      args.show_score_thr, early_stop=args.early_stop)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False)
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                     args.gpu_collect)

        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                eval_kwargs = cfg.get('evaluation', {}).copy()
                # hard-code way to remove EvalHook args
                for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                    eval_kwargs.pop(key, None)
                eval_kwargs.update(dict(metric=args.eval, **kwargs))
                # print(dataset.evaluate(outputs, **eval_kwargs))
                eval_results = dataset.evaluate(outputs, **eval_kwargs)
                print(eval_results)
                # ovtb no realization here
                # combined_result = eval_results['combined_result']
                # base_result = eval_results['base_result']
                # novel_result = eval_results['novel_result']
                # epoch_line = f'{cfg.text}\n {args}\n\n\n'+ '\n' + combined_result+base_result+novel_result + '\n'
            # write_to_file(os.path.join(checkpoint_dir, 'single_eval_result.txt'), epoch_line)
    else:
        # record the new cfg files

        rank, _ = get_dist_info()
        if rank == 0:
            current_time = datetime.datetime.now()
            saved_cfg_file = os.path.join(args.checkpoint_dir, f"{os.path.split(cfg.filename)[-1].split('.')[0]}_{current_time.strftime('%Y_%m_%d_%H_%M_%S') }.py")
            saved_log_file = os.path.join(args.checkpoint_dir, 'eval_result.txt')

            with open(saved_cfg_file, 'w') as f:
                cfg.dist_params['timeout'] = 720000
                f.write(cfg.pretty_text)
            with open(saved_log_file, 'a') as f:
                f.write('\n')
                f.write(saved_cfg_file)
                f.write('\n')

        checkpoint_files = sorted([os.path.join(args.checkpoint_dir, file) for file in os.listdir(args.checkpoint_dir) if 'epoch' in file and os.path.isfile(os.path.join(args.checkpoint_dir, file))], key=lambda f : int(f.split('_')[-1].split('.')[0]))
        origianl_resfile_path = args.eval_options['resfile_path']
        eval_res_list = []
        for checkpoint_index, checkpoint in enumerate(checkpoint_files):
            if args.checkpoint_l_index != -1 and args.checkpoint_r_index != -1:
                if  args.checkpoint_l_index <= checkpoint_index  <= args.checkpoint_r_index:
                    pass
                else:
                    continue
            epoch = int(checkpoint.split('_')[-1].split('.')[0])
            args.checkpoint = checkpoint
            args.eval_options['resfile_path'] = os.path.join(origianl_resfile_path, f'epoch_{epoch}')
            eval_res_list.append(args.eval_options['resfile_path'])
            # build the model and load checkpoint
            model = build_model(cfg.model, train_cfg=None, test_cfg=None)
            model.roi_head.result_save_path = args.eval_options['resfile_path']
            # fp16_cfg = cfg.get('fp16', None)
            # if fp16_cfg is not None:
            #     wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

            if args.fuse_conv_bn:
                model = fuse_conv_bn(model)


            model.CLASSES = dataset.CLASSES

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                          args.show_score_thr, early_stop=args.early_stop)
            else:
                model = MMDistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=False)
                outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                         args.gpu_collect)

            rank, _ = get_dist_info()
            if rank == 0:
                if args.out:
                    print(f'\nwriting results to {args.out}')
                    mmcv.dump(outputs, args.out)
                kwargs = {} if args.eval_options is None else args.eval_options
                if args.format_only:
                    dataset.format_results(outputs, **kwargs)
                    continue
                if args.eval:
                    eval_kwargs = cfg.get('evaluation', {}).copy()
                    # hard-code way to remove EvalHook args
                    for key in ['interval', 'tmpdir', 'start', 'gpu_collect']:
                        eval_kwargs.pop(key, None)
                    eval_kwargs.update(dict(metric=args.eval, **kwargs))
                    eval_results = dataset.evaluate(outputs, **eval_kwargs)
                    print(eval_results)
                    combined_result = eval_results['combined_result']
                    base_result = eval_results['base_result']
                    novel_result = eval_results['novel_result']
                    epoch_line = f'epoch[{epoch}]:'+ '\n' + combined_result+base_result+novel_result + '\n'
                write_to_file(os.path.join(args.checkpoint_dir, 'eval_result.txt'), epoch_line)
                # with open(os.path.join(args.checkpoint_dir, 'eval_result.txt'), 'a') as f:
                #     f.write(epoch_line)
        if args.off_eval and rank == 0:
            val_dataset_path = dataset.ann_file
            off_eval_by_multi_thread(eval_res_list, args.checkpoint_dir, val_dataset_path)
        print()

if __name__ == '__main__':
    main()
