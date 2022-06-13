# from mmdet.apis import init_detector, inference_detector
#
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# device = 'cuda:0'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# inference_detector(model, 'demo/demo.jpg')

# import json
# a = {'132': [[10, 12.213, 132156.213]], '456': [[10, 12, 0.213456]]}
# with open('123.json', 'w') as fp:
#     json.dump(a, fp, indent=2)

# import numpy as np
# polygon = np.array([
#     [510.66, 423.01, 511.72, 420.03, 510.45, 416.0, 510.34, 413.02, 510.77, 410.26, 510.77, 407.5, 510.34, 405.16,
#      511.51, 402.83, 511.41, 400.49, 510.24, 398.16, 509.39, 397.31, 504.61, 399.22, 502.17, 399.64, 500.89, 401.66,
#      500.47, 402.08, 499.09, 401.87, 495.79, 401.98, 490.59, 401.77, 488.79, 401.77, 485.39, 398.58, 483.9, 397.31,
#      481.56, 396.35, 478.48, 395.93, 476.68, 396.03, 475.4, 396.77, 473.92, 398.79, 473.28, 399.96, 473.49, 401.87,
#      474.56, 403.47, 473.07, 405.59, 473.39, 407.71, 476.68, 409.41, 479.23, 409.73, 481.56, 410.69, 480.4, 411.85,
#      481.35, 414.93, 479.86, 418.65, 477.32, 420.03, 476.04, 422.58, 479.02, 422.58, 480.29, 423.01, 483.79, 419.93,
#      486.66, 416.21, 490.06, 415.57, 492.18, 416.85, 491.65, 420.24, 492.82, 422.9, 493.56, 424.39, 496.43, 424.6,
#      498.02, 423.01, 498.13, 421.31, 497.07, 420.03, 497.07, 415.15, 496.33, 414.51, 501.1, 411.96, 502.06, 411.32,
#      503.02, 415.04, 503.33, 418.12, 501.1, 420.24, 498.98, 421.63, 500.47, 424.39, 505.03, 423.32, 506.2, 421.31,
#      507.69, 419.5, 506.31, 423.32, 510.03, 423.01, 510.45, 423.01]])
# polygon = polygon.reshape(-1, 2).tolist()
# print(type(polygon))
# from shapely.geometry import Polygon
# pgon = Polygon(polygon) # Assuming the OP's x,y coordinates
#
# print(pgon.area)
# area = 702.1057499999998
# bbox = [473.07,395.93,38.65,28.67]
# print(bbox[-2] * bbox[-1])
# print(area)
from ensemble_boxes import weighted_boxes_fusion
import numpy as np
# predictions = [
#     [
#       752,
#       56,
#       1148,
#       314,
#       0.99988
#     ],
#     [
#       788,
#       587,
#       1043,
#       795,
#       0.99986
#     ],
#     [
#       1324,
#       162,
#       1444,
#       345,
#       0.99973
#     ],
#     [
#       675,
#       361,
#       821,
#       442,
#       0.99973
#     ],
#     [
#       1042,
#       727,
#       1109,
#       808,
#       0.99503
#     ],
#     [
#       1032,
#       662,
#       1101,
#       733,
#       0.97632
#     ],
#     [
#       1033,
#       661,
#       1102,
#       747,
#       0.12527
#     ],
#     [
#       1039,
#       719,
#       1109,
#       808,
#       0.09354
#     ],
#     [
#       209,
#       302,
#       319,
#       380,
#       0.07381
#     ],
#     [
#       677,
#       362,
#       820,
#       441,
#       0.05251
#     ]
#   ]
#
import cv2
from ensemble_boxes import *
import json
from pathlib import Path


def transform_result(result):
    assert len(result) == 5
    result = list(map(round, result[:-1])) + [round(result[-1].item(), 5)]
    # print(result)
    return result

def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gen_color_list(model_num, labels_num):
    color_list = np.zeros((model_num, labels_num, 3))
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    total = 0
    for i in range(model_num):
        for j in range(labels_num):
            color_list[i, j, :] = colors_to_use[total]
            total = (total + 1) % len(colors_to_use)
    return color_list


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    width = 858
    height = 471
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[...] = 255
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(width * boxes_list[i][j][0])
            y1 = int(height * boxes_list[i][j][1])
            x2 = int(width * boxes_list[i][j][2])
            y2 = int(height * boxes_list[i][j][3])
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), color_list[i][int(lbl)], int(thickness * scores_list[i][j]))
    show_image(image)


def WBF(predictions, weights, iou_thr=0.95, width=1716, height=942):

    bbox_list = []
    scores_list = []
    labels_list = []
    for prediction in predictions:
        bbox = (prediction[:, :-1] / [width, height, width, height]).tolist()
        bbox_list.append(bbox)
        scores = prediction[:, -1:].flatten().tolist()
        scores_list.append(scores)
        labels_list.append(np.zeros_like(scores).tolist())
    # bbox_list = [predictions[:, :-1] / [width, height, width, height]]
    #
    # scores_list = predictions[:, -1:].reshape((1, -1))
    #
    # labels_list = np.zeros_like(scores_list)
    # print(bbox_list)
    # print(scores_list)
    # print(labels_list)
    # exit()
    #
    # show_boxes(bbox_list, scores_list, labels_list)
    #
    boxes, scores, labels = weighted_boxes_fusion(bbox_list, scores_list, labels_list, weights=None,
                                                  iou_thr=iou_thr, skip_box_thr=0.0)
    #
    # show_boxes([boxes], [scores], [labels.astype(np.int32)])
    #
    assert len(boxes) == len(scores)
    boxes = boxes * [width, height, width, height]
    result = np.concatenate((boxes, scores[:, np.newaxis]), axis=1)
    instance_results = result[result[:, -1] >= 0.05].copy()
    instance_results = list(map(transform_result, instance_results))
    # print(instance_results)
    # exit()
    return instance_results


if __name__ == '__main__':
    base_dir = Path('results')
    pred_file_names = ['v3', 'v5', 'custom_v14-6', 'yun']
    # 0.930 0.928 0.934 0.938
    pred_weights = [2, 1, 5, 8]
    pred_file_paths = [base_dir / pred_file_name / 'result.json' for pred_file_name in pred_file_names]
    predictions = []
    output_path = base_dir / 'fusion_result.json'
    fusion_predictions = {}
    for pred_file_path in pred_file_paths:
        with pred_file_path.open('r') as json_file:
            prediction = json.load(json_file)
        predictions.append(prediction)
    test_file_names = sorted(predictions[0].keys())
    for test_file_name in test_file_names:
        print(test_file_name)
        results = []
        for prediction in predictions:
            results.append(np.array(prediction[test_file_name]))
        results = WBF(results, pred_weights)
        fusion_predictions[test_file_name] = results
    with output_path.open('w') as json_file:
        json.dump(fusion_predictions, json_file, indent=2)
