model = dict(
    type='OVTrack',
    freeze_detector=True,
    method='ovtrack-teta',
    uncertainty_ovtrack=True,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
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
        only_use_n_layers=-1,
        anchor_generator=dict(
            type='AnchorGenerator',
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
        type='OVTrackRoIHeadUncertainty',
        finetune_track=True,
        prompt_path='saved_models/pretrained_models/detpro_prompt.pt',
        debug=False,
        only_validation_categories=True,
        use_special_prompt=False,
        use_special_text_prompt=True,
        use_special_image_prompt=True,
        use_special_prompt_only_on_novel=False,
        prompt_word_list=['complete', 'incomplete'],
        init_track_head_by_bbox_head=False,
        self_train=False,
        only_self_train=True,
        use_cls_static_ratio=True,
        cls_static_ratio=0.0,
        use_motion_static_ratio=True,
        motion_static_ratio=0.0,
        feature_fusion_head=dict(
            type='FeatureFusionModule', feature_dim=256, max_fusion_ratio=2.0),
        use_cyc_loss=False,
        cyc_loss_start_iteration=1000,
        self_train_rcnn=dict(
            score_thr=0.0001,
            nms=dict(
                type='nms',
                iou_threshold=0.5,
                class_agnostic=True,
                split_thr=1000000),
            max_per_img=5),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
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
            in_channels=256,
            fc_out_channels=1024,
            num_convs=4,
            num_fcs=1,
            embed_channels=256,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cyc=dict(
                type='CycleLoss', margin=0.5, loss_type=['pairwise']),
            loss_track=dict(
                type='MultiPosCrossEntropyLoss',
                loss_weight=0.25,
                version='unbiased'),
            loss_track_aux=dict(
                type='L2Loss',
                neg_pos_ub=3,
                pos_margin=0,
                neg_margin=0.1,
                hard_mining=True,
                loss_weight=1.0)),
        cyc_loss_ratio=0.01),
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
                neg_pos_ub=10,
                add_gt_as_proposals=True,
                pos_sampler=dict(type='InstanceBalancedPosSampler'),
                neg_sampler=dict(type='RandomSampler')))),
    tracker=dict(
        type='OVTrackerUncertainty',
        init_score_thr=0.0001,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        memo_frames=10,
        momentum_embed=0.8,
        momentum_obj_score=0.5,
        match_metric='bisoftmax',
        match_with_cosine=True,
        contrastive_thr=0.5,
        confused_features=True),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0001,
            nms=dict(
                type='nms',
                iou_threshold=0.5,
                class_agnostic=True,
                split_thr=1000000),
            max_per_img=50)))
dataset_type = 'TaoDataset'
data_root = 'data/tao/'
img_scale = (800, 1333)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_inner_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotationsWithTrack', with_bbox=True, with_track=True)
]
train_outer_pipeline = [
    dict(type='SeqYOLOXHSVRandomAug'),
    dict(
        type='SeqResize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        share_params=True,
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(
        type='SeqNormalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='SeqFilterAnnotationsWithTrack',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='MatchInstancesNew', skip_nomatch=True),
    dict(
        type='VideoCollectWithTrack',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices']),
    dict(type='SeqDefaultFormatBundleWithTrack', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=2,
    persistent_workers=True,
    train=dict(
        type='SeqMultiImageMixDataset',
        dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.001,
            dataset=dict(
                type='TaoDataset',
                classes='data/lvis/annotations/lvis_classes_v1.txt',
                img_prefix='data/tao/frames/',
                load_as_video=True,
                ann_file=
                'saved_models/ctao_dataset/ctao_base.json',
                key_img_sampler=dict(interval=1),
                ref_img_sampler=dict(
                    num_ref_imgs=1, scope=30, method='uniform', pesudo=False),
                is_select_ori_img=True,
                extra_sample_ratio=8.0,
                pipeline=[
                    dict(type='LoadMultiImagesFromFile'),
                    dict(
                        type='SeqLoadAnnotationsWithTrack',
                        with_bbox=True,
                        with_track=True)
                ])),
        pipeline=[
            dict(type='SeqYOLOXHSVRandomAug'),
            dict(
                type='SeqResize',
                img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                           (1333, 768), (1333, 800)],
                share_params=True,
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
            dict(
                type='SeqNormalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='SeqPad', size_divisor=32),
            dict(
                type='SeqFilterAnnotationsWithTrack',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='MatchInstancesNew', skip_nomatch=True),
            dict(
                type='VideoCollectWithTrack',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices']),
            dict(type='SeqDefaultFormatBundleWithTrack', ref_prefix='ref')
        ]),
    val=dict(
        type='TaoDataset',
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]),
    test=dict(
        type='TaoDataset',
        classes='data/lvis/annotations/lvis_classes_v1.txt',
        ann_file='data/tao/annotations/validation_ours_v1.json',
        img_prefix='data/tao/frames/',
        ref_img_sampler=None,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='VideoCollect', keys=['img'])
                ])
        ]))
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[3, 5, 10])
total_epochs = 10
load_from = 'saved_models/pretrained_models/ovtrack_pair.pth'
evaluation = dict(
    metric=['track'],
    start=9999,
    interval=1,
    resfile_path=
    'results/ctao_results')
checkpoint_config = dict(interval=1, create_symlink=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl', timeout=72000)
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
find_unused_parameters = True
work_dir = 'work_dirs/VOVTrack_after/c_tao_training_tidy'
