import argparse
import os
import time
import glob
import math
import torch
import json

import cv2
print(cv2.__version__)
import matplotlib
import numpy as np
import torch
import torch.optim as optim
from PIL import Image as PILImage
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.coco import CocoTrainDataset
from datasets.transformations import (ConvertKeypoints, CropPad, Flip,
                                               Rotate, Scale)
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.get_parameters import (get_parameters_bn,
                                             get_parameters_conv,
                                             get_parameters_conv_depthwise)
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_from_mobilenet, load_state
from modules.loss import l2_loss
from modules.pose import Pose, track_poses
from matplotlib import pyplot as plt
from val import infer, convert_to_coco_format

from demo import ImageReader, VideoReader

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

def run_demo_video(net, frame_provider, device):

    index_frame = 0
    coco_result = []
    previous_poses = []
    current_poses = []
    for frame_id, image in enumerate(frame_provider):
        if frame_id != 0:
            previous_poses = current_poses
        
        image_original = image.copy()

        size = image.shape[1:]

        base_height = 368
        scales = [1]
        stride = 8

        avg_heatmaps, avg_pafs, pafs = infer(net, image_original, scales, base_height, stride)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = 6

        delta_y = size[0] / pafs.shape[0]
        delta_x = size[1] / pafs.shape[1]

        for kpt_idx in range(num_keypoints):  
            C = avg_heatmaps[:, :, kpt_idx]
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        conf = {
            "defined_success_ratio": 0.8,
            "point_score": 100
        }

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs, demo=True, **conf)

        # Draw keypoints
        for index, entry_id in enumerate(all_keypoints_by_type):
            for keypoint in entry_id:
                x, y, _, _ = keypoint
                if x > 2:
                    cv2.putText(image_original, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(image_original, (x, y), 3, (255, 0, 255), -1)
        
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
        
        if frame_id != 0:
            track_poses(previous_poses, current_poses)

        bounding_boxes = []

        # Draw tracking ID and Pose
        for pose in current_poses:
            pose.draw(image_original)
            # cv2.rectangle(image_original, (pose.bbox[0], pose.bbox[1]), (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), color=(255, 0, 0))
            if frame_id != 0:
                cv2.putText(image_original, "id_" + str(pose.id), (pose.keypoints[3][0], pose.keypoints[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #bounding_boxes.append(pose.bbox)
        
        # cv2.imshow("Smt", image_original)
        # cv2.waitKey()
        #plt.show()

        """coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

        index_frame = index_frame + 1
        print(index_frame)
        image_id = "frame_" + str(index_frame)
        for idx in range(len(coco_keypoints)):
            coco_result.append({
                'image_id': image_id,
                'category_id': 1,  # person
                'keypoints': coco_keypoints[idx],
                'bounding_box': bounding_boxes[idx],
                'score': scores[idx]
            })
        
        with open("videos/video_demo_detections.json", 'w') as f:
            json.dump(coco_result, f, indent=4)"""
        index_frame = index_frame + 1
        if index_frame < 10:
            img_name = "000{}".format(index_frame)
        elif index_frame < 100:
            img_name = "00{}".format(index_frame)
        else:
            img_name = "0{}".format(index_frame)

        cv2.imwrite("//home//user1//Desktop//lightweight//temp_id//{}.jpg".format(img_name),image_original)
        


if __name__ == '__main__':
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--video', type=str, required=True, help='path to the video')
    #args = parser.parse_args()
    #video_path = "videos/run.mp4"
    video_filenames = glob.glob("videos/DemoFrames/"+ '*.jpg')
    video_filenames = sorted(video_filenames)
    model_path = "/home/user1/Desktop/lightweight/default_checkpoints/checkpoint_iter_8200.pth"
    frame_provider = ImageReader(video_filenames)

    KEY_POINTS = 6
    net = PoseEstimationWithMobileNet(num_refinement_stages = 1, num_heatmaps = (KEY_POINTS + 1), num_pafs = ((KEY_POINTS - 1) * 2))
    load_state(net, torch.load(model_path))

    device = torch.device("cuda:0")
    if torch.cuda.is_available():
        net = net.cuda().to(device)

    run_demo_video(net, frame_provider, device)