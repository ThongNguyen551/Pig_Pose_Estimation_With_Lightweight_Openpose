import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

from datasets.coco import CocoValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose

def heatmap2d(arr: np.ndarray, image_constant, id_of_paf):
    plt.imshow(image_constant)
    plt.imshow(arr, cmap='viridis', alpha = 0.6)
    plt.title(str(id_of_paf))
    plt.colorbar()
    plt.show()

def run_coco_eval(gt_file_path, dt_file_path):
    annotation_type = 'keypoints'
    print('Running test for {} results.'.format(annotation_type))

    coco_gt = COCO(gt_file_path)
    coco_dt = coco_gt.loadRes(dt_file_path)

    result = COCOeval(coco_gt, coco_dt, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad


def convert_to_coco_format(pose_entries, all_keypoints):
    coco_keypoints = []
    scores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        keypoints = [0] * 17 * 3
        # to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        to_coco_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        person_score = pose_entries[n][-2]
        position_id = -1
        for keypoint_id in pose_entries[n][:-3]:
            position_id += 1
            #if position_id == 1:  # no 'neck' in COCO
            #    continue

            cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
            if keypoint_id != -1:
                cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                cx = cx + 0.5
                cy = cy + 0.5
                visibility = 1
            keypoints[to_coco_map[position_id] * 3 + 0] = cx
            keypoints[to_coco_map[position_id] * 3 + 1] = cy
            keypoints[to_coco_map[position_id] * 3 + 2] = visibility
        coco_keypoints.append(keypoints)
        scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
    return coco_keypoints, scores


def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, 7), dtype=np.float32) # 7
    avg_pafs = np.zeros((height, width, 10), dtype=np.float32) # 10

    for ratio in scales_ratios:
        scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        min_dims = [base_height, max(scaled_img.shape[1], base_height)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
        pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_pafs = avg_pafs + pafs / len(scales_ratios)

    return avg_heatmaps, avg_pafs, pafs


def evaluate(labels, output_name, images_folder, net, multiscale=False, visualize=False):
    net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = CocoValDataset(labels, images_folder)
    coco_result = []
    plot_index = 0
    for sample in dataset:
        file_name = sample['file_name']
        img = sample['img']
        image_constant = img
        size = img.shape[1:]

        avg_heatmaps, avg_pafs, pafs = infer(net, img, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(6):  # 7th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs, image_constant)

        delta_y = size[0] / pafs.shape[0]
        delta_x = size[1] / pafs.shape[1]

        for index, entry_id in enumerate(all_keypoints_by_type):
            for keypoint in entry_id:
                x, y, _, _ = keypoint

                cv2.putText(img, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (255, 0, 255), -1)
        
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((6, 2), dtype=np.int32) * -1
            for kpt_id in range(6):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        for pose in current_poses:
            pose.draw(img)
        img = cv2.resize(img, (1248, 702))
        cv2.imshow("Figure", img)
        cv2.waitKey()

        #coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        #image_id = int(file_name[0:file_name.rfind('.')])
        """image_id = int(file_name.split(".")[1].split("_")[-1])
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'score': scores[idx]
            })"""
        
        """current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((6, 2), dtype=np.int32) * -1
            for kpt_id in range(6):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)"""

        plot_index += 1
        htmp = np.sum(avg_heatmaps, axis = 2)
        pafm = np.sum(avg_pafs, axis = 2)
        #plt.imshow(htmp)
        #plt.show()
        #plt.imshow(pafm)
        #plt.show()
        
        """id_of_paf = 0
        proba = avg_pafs[:, :, id_of_paf]
        heatmap2d(proba, image_constant, id_of_paf)

        id_of_paf = 1
        proba = avg_pafs[:, :, id_of_paf]
        heatmap2d(proba, image_constant, id_of_paf)"""

        id_of_paf = 8
        proba = avg_pafs[:, :, id_of_paf]
        heatmap2d(proba, image_constant, id_of_paf)

        id_of_paf = 9
        proba = avg_pafs[:, :, id_of_paf]
        heatmap2d(proba, image_constant, id_of_paf)
        """fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.title.set_text('Heatmap ' + str(plot_index))
        ax1.imshow(htmp, cmap='RdYlBu')
        im1 = ax1.imshow(image_constant, alpha = 0.4)

        fig.colorbar(im1, ax=ax1, orientation='vertical')

        ax2.title.set_text('PAF' + str(plot_index))
        ax2.imshow(pafm, cmap='RdYlBu')
        im2 = ax2.imshow(image_constant, alpha = 0.4)

        fig.colorbar(im2, ax=ax2, orientation='vertical')
        plt.show()"""
        #fig.savefig('/home/user1/Desktop/plots/' + str(plot_index) + 'plots.png')
        
        """

        visualize = True
        if visualize:
            for keypoints in coco_keypoints:
                for idx in range(len(keypoints) // 3):
                    x = int(keypoints[idx * 3]) + 10
                    y = int(keypoints[idx * 3 + 1]) + 10
                    cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                               3, (255, 0, 255), -1)
            cv2.imwrite('/home/user1/Desktop/result/' + str(plot_index) + 'image.png', img)
            #key = cv2.waitKey()
            #if key == 27:  # esc
                #return"""

    """with open(output_name, 'w') as f:
        json.dump(coco_result, f, indent=4)

    run_coco_eval(labels, output_name)"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', type=str, required=True, help='path to json with keypoints val labels')
    parser.add_argument('--output-name', type=str, default='detections.json',
                        help='name of output json file with detected keypoints')
    parser.add_argument('--images-folder', type=str, required=True, help='path to COCO val images folder')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--multiscale', action='store_true', help='average inference results over multiple scales')
    parser.add_argument('--visualize', action='store_true', help='show keypoints')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)

    evaluate(args.labels, args.output_name, args.images_folder, net, args.multiscale, args.visualize)
