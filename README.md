# AI-CUP Competition: STAS Detection
## Environment Setup
Device: single 2080ti with CUDA 10.2 and python3.8
```bash
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
python setup.py install
# python setup.py develop
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

### Apex Installation:
Following [Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection), we use apex for mixed precision training by default. To install apex, run:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

### Fix Apex's Bug:
After intalling apex, we should slightly modify the source code of Apex to fix the bug before executing the codes. Take the situation of using virtual environment as example. Goto your environment folder(eg. venv), modify venv/lib/python3.8/site-packages/apex/amp/utils.py line 97
```python 
-   if cached_x.grad_fn.next_functions[1][0].variable is not x:       
+   if cached_x.grad_fn.next_functions[0][0].variable is not x:
        raise RuntimeError("x and cache[x] both require grad, but x is not "
                                   "cache[x]'s parent.  This is likely an error.")
```

## Preparation before executing the code
### Model Checkpoint and Config file
The model we use to finetune can be found on [the repository of CBNetV2](https://github.com/VDIGPKU/CBNetV2).
We recommend to unzip and put pretrained pth model in [**ckpt** folder](ckpt/), or you need to modify the line 256. 
Before start training, make sure that the folder contains the data is correctly defined on the line 265 of the configuration file.
```python=265
data_root = 'ckpt/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth/'
```
>[origin pretrained model](https://github.com/CBNetwork/storage/releases/download/v1.0.0/htc_cbv2_swin_base22k_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.pth.zip)   
>[origin config](https://github.com/VDIGPKU/CBNetV2/blob/main/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_giou_4conv1f_adamw_20e_coco.py)     
   
unzip competition models and configs folder and name it **work_dirs** (work_dirs)
You need to complete all previous training or download the competition model, put the checkpoint file to the work_dirs folder and name it AI-CUP_semantic_segmentation.pth
>[competition models checkpoints](https://www.dropbox.com/s/kovxf70m9weul8q/AI-CUP_semantic_segmentation.pth?dl=0)  
>[competition segmentation config](configs/cbnet/AI_CUP_SEG.py)

### Data and Annotations
Put data and annotations in [**data/SEG_Train_Datasets**](data/SEG_Train_Datasets) folder, annotations should contain the file from both of the AI-CUP competition of detection and segmentation. 
```
data
`-- SEG_Train_Datasets
    |-- Test_Images
    |   `-- test_images...
    |-- Train_Annotations
    |   |-- json_segmentation_annotations...
    |   `-- xml_detection_annotations...
    `-- Train_Images
        `-- train_images...
```

Then run **[python convert_STAS.py](python convert_STAS.py)**. The data folder is defined on the line 245 of this python script. you can modify it if needed. After executing the script, **coco** and **custom** folder should appear in the directory
```
data
`-- SEG_Train_Datasets
    |-- same_with_above...
    |-- coco
    |   |-- STAS_final.json
    |   |-- STAS_test.json
    |   |-- STAS_train.json
    |   `-- STAS_val.json
    `-- custom
        |-- STAS_final.pkl
        |-- STAS_test.pkl
        |-- STAS_train.pkl
        `-- STAS_val.pkl
```
#### COCO Format For Segmentation Annotations (json)

```python
'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # 如果有 mask 標籤
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
```

#### Middle Format For Detection Annotations (pickle)
```python

[
    {
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, ) （可選字段）
        }
    },
    ...
]
```

### Configuration file
The configuration file we use in this competition is **[configs/cbnet/AI_CUP_SEG.py](configs/cbnet/AI_CUP_SEG.py)**. 
Before start training, make sure that the folder contains the data is correctly defined on the line 242 of the configuration file.
```python=242
data_root = 'data/SEG_Train_Datasets'
```

## How to Train
**Please only use a single GPU for train**     
```bash 
python -m torch.distributed.launch tools/train.py \
    configs/cbnet/AI_CUP_SEG.py \
    --gpus 1 --deterministic --seed 123 \
    --work-dir work_dirs
```

## How to Evaluate
**Please only use a single GPU for inference**    

```bash
# You need to complete all previous training or download the competition model, put the checkpoint file to the work_dirs folder and name it AI-CUP_semantic_segmentation.pth
python tools/test.py \
    configs/cbnet/AI_CUP_SEG.py \
    work_dirs/AI-CUP_semantic_segmentation.pth \
    --show --show-dir output \
    --show-score-thr 0.05 --coco
```
You can modify **--show-dir** parameter to change the output folder. If you use the setting above, the result would be store at the /output/seg_result/

## Other Links
> **Original CBNet**: See [CBNet: A Novel Composite Backbone Network Architecture for Object Detection](https://github.com/VDIGPKU/CBNet).    
> **Origin CBNetV2 Github**: See [VDIGPKU CBNetV2](https://github.com/VDIGPKU/CBNetV2)
## Citation
If you use our code/model, please consider to cite our paper [CBNetV2: A Novel Composite Backbone Network Architecture for Object Detection](http://arxiv.org/abs/2107.00420).
```
@article{liang2021cbnetv2,
  title={CBNetV2: A Composite Backbone Network Architecture for Object Detection}, 
  author={Tingting Liang and Xiaojie Chu and Yudong Liu and Yongtao Wang and Zhi Tang and Wei Chu and Jingdong Chen and Haibing Ling},
  journal={arXiv preprint arXiv:2107.00420},
  year={2021}
}
```
