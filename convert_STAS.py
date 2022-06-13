import json
from pathlib import Path
import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from argparse import ArgumentParser
from shapely.geometry import Polygon


def convert_bbox_to_coco(box):
    x = box[0]
    y = box[1]
    w = box[2] - box[0]
    h = box[3] - box[1]
    return [x, y, w, h]


def parse_xml_to_json(image_path):
    # print(anno_dir)
    image_name = anno_dir / f'{image_path.stem}.xml'
    # print(image_name)
    # print(type(image_name))
    with image_name.open(encoding='utf-8') as xml_file:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bboxes = []
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text))
            # bboxes.append(convert_bbox(box))
            bboxes.append(box)
        bboxes = np.array(bboxes, dtype=np.float32)
        annotation = {
            'filename': str(image_path),
            'width': w,
            'height': h,
            'ann': {
                'bboxes': np.array(bboxes, dtype=np.float32),
                'labels': np.zeros(bboxes.shape[0], dtype=np.int64),
            }
        }
    return annotation


def parse_seg_to_coco(image_path):
    image_name = anno_dir / f'{image_path.stem}.json'
    with image_name.open(encoding='utf-8') as json_file:
        annotation = json.load(json_file)
        instance_annos = []
        file_name = annotation['imagePath']
        image_id = os.path.splitext(file_name)[0]
        meta_image = {
            'file_name': file_name,
            'height': annotation['imageHeight'],
            'width': annotation['imageWidth'],
            'id': image_id
        }
        polygon_shapes = annotation['shapes']
        for polygon_shape in polygon_shapes:
            polygon = np.array(polygon_shape['points'])
            x_min = polygon[:, 0].min()
            x_max = polygon[:, 0].max()
            y_min = polygon[:, 1].min()
            y_max = polygon[:, 1].max()
            area = Polygon(polygon).area
            instance_anno = {
                'segmentation': [polygon.flatten().tolist()],
                'area': area,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'category_id': 0
            }
            instance_annos.append(instance_anno)

    return meta_image, instance_annos


def convert_to_middle_format():
    custom_base_dir = base_dir / 'custom'
    custom_base_dir.mkdir(parents=True, exist_ok=True)

    final_train_output_path = custom_base_dir / 'STAS_final.pkl'
    train_output_path = custom_base_dir / 'STAS_train.pkl'
    val_output_path = custom_base_dir / 'STAS_val.pkl'
    test_output_path = custom_base_dir / 'STAS_test.pkl'

    annotations = []
    for image_path in train_dir.iterdir():
        annotations.append(parse_xml_to_json(image_path))
    random.shuffle(annotations)
    # print(annotations)
    train_annotations = annotations[:int(len(annotations) * args.split_ratio)]
    val_annotations = annotations[int(len(annotations) * args.split_ratio):]
    with final_train_output_path.open('wb') as f:
        pickle.dump(annotations, f)
    with train_output_path.open('wb') as f:
        pickle.dump(train_annotations, f)
    with val_output_path.open('wb') as f:
        pickle.dump(val_annotations, f)

    #     for test
    test_annotations = []
    for image_path in test_dir.iterdir():
        test_annotations.append({'filename': str(image_path)})
    with test_output_path.open('wb') as f:
        pickle.dump(test_annotations, f)


def convert_to_coco_format():
    coco_base_dir = base_dir / 'coco'
    coco_base_dir.mkdir(parents=True, exist_ok=True)

    final_train_output_path = coco_base_dir / 'STAS_final.json'
    train_output_path = coco_base_dir / 'STAS_train.json'
    val_output_path = coco_base_dir / 'STAS_val.json'
    test_output_path = coco_base_dir / 'STAS_test.json'
    # TODO new
    file_path = [train_dir / image_name for image_name in os.listdir(train_dir)]
    random.shuffle(file_path)
    train_file_path = file_path[:int(len(file_path) * args.split_ratio)]
    val_file_path = file_path[int(len(file_path) * args.split_ratio):]
    train_images = []
    val_images = []
    train_annotations = []
    val_annotations = []
    anno_id = 0
    for image_path in train_file_path:
        meta_image, instance_annos = parse_seg_to_coco(image_path)
        train_images.append(meta_image)
        for instance_anno in instance_annos:
            instance_anno['id'] = anno_id
            anno_id += 1
        train_annotations += instance_annos
    for image_path in val_file_path:
        meta_image, instance_annos = parse_seg_to_coco(image_path)
        val_images.append(meta_image)
        for instance_anno in instance_annos:
            instance_anno['id'] = anno_id
            anno_id += 1
        val_annotations += instance_annos
    print(len(train_annotations))
    print(len(val_annotations))
    print(len(val_annotations + train_annotations))
    with train_output_path.open('w') as f:
        train_output = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': [{'id': 0, 'name': 'stas'}]
        }
        json.dump(train_output, f)
    with val_output_path.open('w') as f:
        val_output = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': [{'id': 0, 'name': 'stas'}]
        }
        json.dump(val_output, f)
    with final_train_output_path.open('w') as f:
        final_output = {
            'images': train_images + val_images,
            'annotations': train_annotations + val_annotations,
            'categories': [{'id': 0, 'name': 'stas'}]
        }
        json.dump(final_output, f)
    # TODO new
    # train_images = []
    # annotations = []
    # anno_id = 0
    # for image_path in train_dir.iterdir():
    #     meta_image, instance_annos = parse_seg_to_coco(image_path)
    #     train_images.append(meta_image)
    #     for instance_anno in instance_annos:
    #         instance_anno['id'] = anno_id
    #         anno_id += 1
    #     annotations += instance_annos
    # random.shuffle(annotations)
    # train_annotations = annotations[:int(len(annotations) * args.split_ratio)]
    # val_annotations = annotations[int(len(annotations) * args.split_ratio):]
    # with final_train_output_path.open('w') as f:
    #     final_output = {
    #         'images': train_images,
    #         'annotations': annotations,
    #         'categories': [{'id': 0, 'name': 'stas'}]
    #     }
    #     json.dump(final_output, f)
    # with train_output_path.open('w') as f:
    #     train_output = {
    #         'images': train_images,
    #         'annotations': train_annotations,
    #         'categories': [{'id': 0, 'name': 'stas'}]
    #     }
    #     json.dump(train_output, f)
    # with val_output_path.open('w') as f:
    #     val_output = {
    #         'images': train_images,
    #         'annotations': val_annotations,
    #         'categories': [{'id': 0, 'name': 'stas'}]
    #     }
    #     json.dump(val_output, f)

    #     for test
    test_images = []
    for image_path in test_dir.iterdir():
        test_image_name = image_path.name
        test_images.append({
                'file_name': test_image_name,
                'id': os.path.splitext(test_image_name)[0]
            })
    with test_output_path.open('w') as f:
        test_output = {
            'images': test_images,
            'categories': [{'id': 0, 'name': 'stas'}]
        }
        json.dump(test_output, f)


def parse_args():
    parser = ArgumentParser()

    # data
    parser.add_argument("--split_ratio", type=float, default=0.8)

    args = parser.parse_args()
    return args


def main():
    random.seed(123)
    convert_to_middle_format()
    convert_to_coco_format()


if __name__ == '__main__':
    args = parse_args()

    base_dir = Path('data') / 'SEG_Train_Datasets'
    train_dir = base_dir / 'Train_Images'
    test_dir = base_dir / 'Test_Images'
    anno_dir = base_dir / 'Train_Annotations'

    main()

'''
{
        'filename': 'a.jpg',
        'width': 1280,
        'height': 720,
        'ann': {
            'bboxes': <np.ndarray, float32> (n, 4),
            'labels': <np.ndarray, int64> (n, ),
            'bboxes_ignore': <np.ndarray, float32> (k, 4),
            'labels_ignore': <np.ndarray, int64> (k, )
        }
},
'''
'''
{
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
                249.06]],  # 如果有 mask 标签
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
}
'''
