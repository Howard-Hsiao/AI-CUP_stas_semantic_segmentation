_base_ = 'htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco.py'
dataset_type = 'CustomDataset'

data_root = 'data/OBJ_Train_Datasets/'

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)
                # loss_bbox=dict(type='FocalCIoULoss', loss_weight=10.0, gamma=0.5)
            ),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)
                # loss_bbox=dict(type='FocalCIoULoss', loss_weight=10.0, gamma=0.5)
            ),

            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                # loss_cls=dict(
                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='CIoULoss', loss_weight=10.0)
                # loss_bbox=dict(type='FocalCIoULoss', loss_weight=10.0, gamma=0.5)
            )
        ]
    )
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 1600 1400
origin_image_size = (942, 1716)
# first version
# mutli_scale_image_size = [(850, 471), (850, 500), (900, 500), (900, 520)]
# new_image_size = (1000, 565)
# new_image_size = (1200, 660)
# second version
# mutli_scale_image_size = [(850, 471), (850, 500), (900, 500), (900, 520)]
# new_image_size = (1716, 942)
# third version
# mutli_scale_image_size = [(686, 376), (686, 420), (850, 471), (850, 500), (900, 500), (900, 520)]
# test_mutli_scale_image_size = [(858, 471), (943, 518), (1000, 565), (1115, 612), (1200, 660)]
# new_image_size = (1000, 565)
# fourth version
# mutli_scale_image_size = [(429, 235), (950, 520)]
# test_mutli_scale_image_size = [(686, 376), (858, 471), (1000, 565), (1200, 660)]
# fifth version
mutli_scale_image_size = [(376, 686), (520, 950)]
test_mutli_scale_image_size = [(471, 858), (518, 943), (565, 1000), (612, 1115), (660, 1200)]
# (1000, 565)
# [(1600, 1000), (1600, 1400), (1800, 1200), (1800, 1600)]

albu_train_transforms = [
    # dict(
    #     type='HorizontalFlip',
    #     p=0.2),
    # dict(
    #     type='VerticalFlip',
    #     p=0.2),

    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=45,
    #     interpolation=1,
    #     p=0.2),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
            dict(type='FancyPCA', alpha=0.1, always_apply=False, p=1.0), #trick
        ],
        p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    #
    # dict(type='ChannelShuffle', p=0.1),
    # dict(type='RandomRotate90', always_apply=False, p=0.5), # 随机旋转
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0),
            dict(type='MotionBlur', blur_limit=6, always_apply=False, p=1.0)#trick
        ],
        p=0.1),
]

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='Resize',
#         img_scale=mutli_scale_image_size,
#         multiscale_mode='range',
#         keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(
#         type='Collect',
#         keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=test_mutli_scale_image_size,
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-4,
# )
samples_per_gpu=1
workers_per_gpu=1
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4*(samples_per_gpu/2), betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[6, 10])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=50)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 10])

classes = ('stas',)
# workflow = [('train', 1), ('val', 1)]
workflow = [('train', 1)]
train_pipeline = [
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.5, 0.7, 0.9),
    #     min_crop_size=0.3),
    # yolox
    dict(type='Mosaic', img_scale=origin_image_size, pad_val=114.0, prob=0.2),
    # dict(
    #     type='RandomAffine',
    #     scaling_ratio_range=(0.5, 1.5),
    #     border=(-origin_image_size[0] // 2, -origin_image_size[1] // 2)),
    dict(
        type='MixUp',
        img_scale=origin_image_size,
        ratio_range=(0.8, 1.6),
        pad_val=114.0, prob=0.2),
    # yolox
    dict(
        type='Resize',
        img_scale=mutli_scale_image_size,
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels']),
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'custom/STAS_final.pkl',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=train_dataset,
    # train=dict(
    #     type=dataset_type,
    #     classes=classes,
    #     ann_file=data_root + 'custom/STAS_train.pkl',
    #     pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'custom/STAS_val.pkl',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'custom/STAS_test.pkl',
        pipeline=test_pipeline))

evaluation = dict(metric=['mAP'])

load_from = 'ckpt/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth'
# CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --master_port 29501 tools/train.py configs/cbnet/swin_custom.py --gpus 1 --deterministic --seed 123  --work-dir work_dirs/swin_custom_v14-9
# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --master_port 29501 tools/train.py configs/cbnet/swin_custom.py --gpus 1 --deterministic --seed 123  --work-dir work_dirs/swin_custom_v14-10
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 29502 tools/train.py configs/cbnet/swin_custom.py --gpus 1 --deterministic --seed 123  --work-dir work_dirs/swin_custom_v10-2
# CUDA_VISIBLE_DEVICES=2,3 tools/dist_train.sh configs/cbnet/swin_custom.py 2
# CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/cbnet/swin_custom.py work_dirs/swin_custom_v14-6/latest.pth --out result.json --show --show-dir ckpt
# CUDA_VISIBLE_DEVICES=3 python tools/test.py configs/cbnet/swin_custom.py work_dirs/swin_custom_v14-8/latest.pth --eval mAP
