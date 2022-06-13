import os.path
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from PIL import Image
from pathlib import Path
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


def WBF(predictions, iou_thr=0.55, width=1716, height=942):
    # print(predictions)
    bbox_list = (predictions[:, :-1] / [width, height, width, height])[np.newaxis, :]

    scores_list = predictions[:, -1:].reshape((1, -1))
    # print(bbox_list.shape)
    # print(scores_list.shape)
    labels_list = np.zeros_like(scores_list)
    boxes, scores, labels = weighted_boxes_fusion(bbox_list, scores_list, labels_list, weights=None,
                                                  iou_thr=iou_thr, skip_box_thr=0.0)
    assert len(boxes) == len(scores)
    boxes = boxes * [width, height, width, height]
    result = np.concatenate((boxes, scores[:, np.newaxis]), axis=1)
    # print(len(predictions))
    # print(len(result))
    return result



def transform_result(result):
    assert len(result) == 5
    result = list(map(round, result[:-1])) + [round(result[-1].item(), 5)]
    # print(result)
    return result


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    eval_metric='mAP',
                    is_coco=False):
    confidence_score_thr = 0.05
    model.eval()
    if eval_metric is None:
        results = {}
    else:
        results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        # WBF
        # for j in range(batch_size):
        #     result[j][0] = WBF(result[j][0].copy())
        # WBF
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                # # TODO transform result block
                if eval_metric is None:
                    # print(result[i][0].shape)
                    filename = os.path.split(img_metas[i]['ori_filename'])[-1]
                    # print(type(result[i]))
                    # print(len(result[i]))
                    # print(result[i][0][0].shape)
                    # print(result[i][1][0][0].shape)
                    if is_coco:
                        bboxes = result[i][0][0]
                        if len(result[i][1][0]) == 0:
                            print(img_metas[i]['ori_filename'])
                            seg_image = np.full((img_metas[i]['ori_shape'][0], img_metas[i]['ori_shape'][1]), False)
                        else:
                            seg_image = np.any(result[i][1][0], axis=0)
                        im = Image.fromarray(seg_image)
                        image_path = Path(out_dir) / 'seg_result'
                        image_path.mkdir(parents=True, exist_ok=True)
                        im.save(image_path / f"{os.path.splitext(filename)[0]}.png")
                        instance_results = bboxes[bboxes[:, -1] >= confidence_score_thr].copy()
                        instance_results = list(map(transform_result, instance_results))
                        results[filename] = instance_results
                        # im = Image.fromarray(seg_image)
                        # image_path = Path(out_dir) / 'seg_result'
                        # image_path.mkdir(parents=True, exist_ok=True)
                        # im.save(image_path / f"{os.path.splitext(filename)[0]}.png")
                    else:
                        instance_results = result[i][0][result[i][0][:, -1] >= confidence_score_thr].copy()
                        instance_results = list(map(transform_result, instance_results))
                        results[filename] = instance_results
                # # TODO transform result block
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, 'det_result', img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        if eval_metric is not None:
            results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
