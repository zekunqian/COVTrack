
model = dict(
    type='OVTrackSelfTrain',
    freeze_detector=True,
    method='ovtrack-tao',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(type='Pretrained',
                      checkpoint='open-mmlab://detectron2/resnet50_caffe')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    rpn_head=dict(
        type='MyRPNHead',
        in_channels=256,
        feat_channels=256,
        # only_use_n_layers=2,
        only_use_n_layers=-1,
        anchor_generator=dict(
            type='AnchorGenerator',
            # scales=[8,16,32,64],
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='OVTrackRoIHead',
        finetune_track=True,
        prompt_path='saved_models/pretrained_models/detpro_prompt.pt',
        # debug mode
        debug=False,
        # cluster mode
        # cluster_num=-1,#cluster_num=8,
        # cluster_num=3,
        # cluster_method='dbscan', #'kmeans',
        cluster_num=8,
        cluster_method='kmeans', #'kmeans',
        # only validation categories
        only_validation_categories=True,
        only_test_categories=False,
        # using special prompt
        use_special_prompt=True,
        use_special_text_prompt=True,
        use_special_image_prompt=True,
        use_special_prompt_only_on_novel=False,
        prompt_word_list=["complete", "incomplete"],
        prompt_group_list=["complete", "incomplete",
                           # "blur", "clear",
                           "occluded", "unoccluded",
                           "obscured", "unobscured",
                           "recognizable", "unrecognizable"],
        prompt_group_mean_way=True,
        # self train part
        init_track_head_by_bbox_head=False,  # dangers force init, if no necessary, remove it !!!
        self_train=True,
        only_self_train=True,
        self_train_rcnn=dict(
            dynamic_rcnn_thre=False,
            # score_thr=0.0001, # dynamic
            score_thr=0.00337037037037037, # dynamic
            # score_thr=0.0043, # dynamic
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True, split_thr=1000000),
            max_per_img=50
        ),

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            #TODO only used to test
            # finest_scale=0.1,
        ),
        bbox_head=dict(
            type='Shared4Conv1FCCliPBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            ensemble=True,
            with_cls=False,
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        track_head=dict(
            type='QuasiDenseEmbedHead',
            # type='QuasiDenseEmbedHeadSharedConvSize', # use the shared conv and fcs and remove fc_embed, which can help load classification region embedding
            in_channels=256,
            fc_out_channels=1024,
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cyc=dict(
                type='CycleLoss',
                margin=0.5,
                loss_type=['pairwise', 'triplewise'],
                # loss_type=['pairwise'],
            ),
            loss_track=dict(type='MultiPosCrossEntropyLoss', loss_weight=0.25,
                            version="unbiased"),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False),
        embed=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='CombinedSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=3,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))
    ),
    tracker=dict(
        type='OVTracker',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        match_metric='bisoftmax',
        match_with_cosine=True,
        contrastive_thr=0.5,
    ),
test_cfg = dict(
    rpn=dict(
        nms_pre=1000,
        max_per_img=1000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),

    # rcnn=dict(
    #     score_thr=0.0001,
    #     nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True, split_thr=1000000),
    #     max_per_img=50)

    rcnn = dict(
        score_thr=0.0001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=300,
        mask_thr_binary=0.5),

    )


)

dataset_type = 'TaoDataset'
data_root = 'data/tao/'

img_scale = (800, 1333)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict( type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=False, with_ins_id=True),

    # dict(type='SeqMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='SeqRandomAffine'),
    dict(type='SeqYOLOXHSVRandomAug'),
    dict(type='SeqRandomFlip',share_params=False, flip_ratio=0.5),

    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=False,
        multiscale_mode='value',
        keep_ratio=True),

    dict(type='SeqNormalize', **img_norm_cfg),
    dict(
        type='SeqPad',
        size_divisor=32,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel
        ),
    # dict(type='SeqFilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='SeqDefaultFormatBundle'),
    dict(
        type='SeqCollectNoPair',
        keys=['img']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadImageFromFile',
    #      file_client_args=dict(
    #          img_db_path='data/tao/tao_val_imgs.hdf5',
    #          backend='hdf5',
    #          type='tao')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

## datasets settings
dataset_type = 'TaoDataset'
tao_train_dataset_list = ['/data1/clark/dataset/openDomain/TAO/tao/frames/train/YFCC100M',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/LaSOT',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/HACS',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/Charades',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/BDD',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/AVA',
                      '/data1/clark/dataset/openDomain/TAO/tao/frames/train/ArgoVerse']
lv_vis_dataset_list = ['/data1/clark/dataset/LV-VIS/train/JPEGImages']
combined_dataset_list = []
combined_dataset_list.extend(tao_train_dataset_list)
combined_dataset_list.extend(lv_vis_dataset_list)


data = dict(
    # samples_per_gpu=3*2-1,
    # workers_per_gpu=4*2-1,
    # samples_per_gpu=3,
    # workers_per_gpu=4,
    samples_per_gpu=4,
    workers_per_gpu=5,
    # samples_per_gpu=12,
    # workers_per_gpu=13,
    # samples_per_gpu=3*2,
    # workers_per_gpu=4*2,
    # samples_per_gpu=3*2,
    # workers_per_gpu=4*2,
    persistent_workers=True,
    train=dict(
            type='CustomVideoDataset',
            # ann_file = '/data1/clark/dataset/LV-VIS/train/JPEGImages',
            ann_file=tao_train_dataset_list,
            # ann_file=combined_dataset_list,
            pipeline=train_pipeline,
            seq_length=8,
            min_seq_length=8,
            crop_size_ratio=8,
            long_short_mode=True,
            long_short_length=16
    ),
    val=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',

        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=test_pipeline)

)

# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='SGD', lr=1e-7, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=1e-7, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 1000,
#     step=[3, 5])
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[3, 6])
    # step = [3, 14])
total_epochs = 20
# load_from = 'saved_models/pretrained_models/ovtrack_clip_distillation.pth'
# load_from = '/home/clark/workspace2/detpro/workdirs/vild_use_complete_to_train_without_complete_test_low_lr/epoch_1.pth'
# load_from = '/home/clark/workspace2/ovtrack/work_dirs/ovtrack_train_by_detpro_complete_finetune_low_lr/epoch_5.pth'

# 1. after softmax best annotation train result (only use complete and incomplete)
# load_from = '/data1/clark/models/ovtrack/resutls/work_dirs/ovtrack_train_by_detpro_epoch18_after_softmax_1e-4/epoch_5.pth'
# 2. best tao finetune result (epoch2 novel better, epoch3 base better)
# load_from = '/home/clark/workspace2/ovtrack/work_dirs/self_train_fintune/ovtrack_after_softmax_e4_train_by_cycAs_TAO/epoch_2.pth'
# 3. after detpro batch prompt annotation training result (4 gruops of prompts)
load_from = '/data1/clark/models/ovtrack/resutls/work_dirs/ovtrack_train_by_detpro_group_mean_finetune/epoch_4.pth'
# 4. batch prompt old 3090 long train
# load_from = '/data1/clark/models/ovtrack/resutls/work_dirs/ovtrack_train_by_detpro_group_mean_finetune_after_long_train/epoch_6.pth'

# load_from = '/home/clark/workspace2/ovtrack/work_dirs/self_train/detpro_low_lr_fintune_self_train_by_tao_dynamic_lre7/epoch_1.pth'
# load_from = '/home/clark/workspace2/detpro/workdirs/vild_ens_20e_fg_bg_5_10_end/epoch_18.pth'
# load_from = '/home/clark/workspace2/detpro/workdirs/vild_use_complete_to_train_without_complete_test_low_lr_6e/epoch_1.pth'
# evaluation = dict(metric=['track'], start=6, interval=1, resfile_path='/scratch/tmp/')
# time_str = str(time.time()).split('.')[0]
evaluation = dict(metric=['track'], start=1, interval=1, resfile_path='results/save_eval_self_train_by_tao_test_fintune_test_longshort_193ud9')
checkpoint_config = dict(interval=1, create_symlink=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
# resume_from ='/data1/clark/models/ovtrack/resutls/work_dirs/self_train_fintune/ovtrack_batch_prompt_e4_by_cycAs_TAO_longshort16_cluster3_epoch4/epoch_1.pth'
# resume_from= '/data1/clark/models/ovtrack/resutls/work_dirs/self_train_fintune/ovtrack_after_softmax_e7_train_by_cycAs_TAO_cluster8_fre/epoch_1.pth'
# 1. batch training strategy only fintune train by TAO
# resume_from = '/home/clark/workspace2/ovtrack/work_dirs/self_train_fintune/ovtrack_batch_prompt_e4_by_cycAs_TAO/epoch_2.pth'
# resume_from = '/data1/clark/models/ovtrack/resutls/work_dirs/self_train_fintune/ovtrack_batch_prompt_e4_by_cycAs_TAO_longshort16_clusterscandb_epoch4/epoch_1.pth'
workflow = [('train', 1)]
gpu_ids = range(0,8)
find_unused_parameters = True
