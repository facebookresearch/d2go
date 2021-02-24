#!/usr/bin/env python3

from detectron2.evaluation import DatasetEvaluator
from detectron2.data import MetadataCatalog
from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm

import torch
import numpy as np

from collections import OrderedDict
import itertools
import json
import logging
import time
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class FaceTrackingKeypointEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name: str, distributed: bool = True):
        self._metadata = MetadataCatalog.get(dataset_name)
        assert hasattr(self._metadata, "json_file"), "No annotation json file given"
        with open(PathManager.get_local_path(self._metadata.json_file), 'r') as f:
            self._json_data = json.load(f)
        self._annotations = self._json_data["annotations"]
        self._image_id_to_annotation = {
            ann["image_id"]: ann for ann in self._annotations
        }
        self._distributed = distributed

        # initialize evaluation metrics
        self._num_keypoints = len(self._metadata.keypoint_names)
        self._pck_thresholds_in_pixels = [1, 5, 10, 20, 40]
        self.reset()

    def reset(self):
        self._gt_kps = []
        self._normalization_factors = []
        self._pred_kps = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            self._normalization_factors.append(
                min(input["width"], input["height"]))
            # pick the instance with highest score as final prediction
            if output["instances"].scores.nelement() != 0:
                output_instance_idx = torch.argmax(output["instances"].scores).item()
                pred_keypoints = output["instances"].pred_keypoints[output_instance_idx, :]
            else:
                pred_keypoints = torch.zeros([self._num_keypoints, 3], dtype=torch.float)

            annotation = self._image_id_to_annotation[input["image_id"]]
            gt_keypoints = torch.Tensor(
                annotation["keypoints"]).to(pred_keypoints.device)
            assert self._num_keypoints * 3 == gt_keypoints.shape[0]

            pred_keypoints = pred_keypoints.cpu().numpy()
            gt_keypoints = torch.reshape(
                gt_keypoints, (self._num_keypoints, 3)).cpu().numpy()

            self._gt_kps.append(gt_keypoints)
            self._pred_kps.append(pred_keypoints)

    def _compute_pcf(
        self, l2: np.ndarray, thresholds: List[int]
    # pyre-fixme[11]: Annotation `float` is not defined as a type.
    ) -> Dict[int, np.float]:
        """
        Computes Percentage of Correct Frames(PCF).
        A frame is correct if all keypoints in
        the frame are within the given threshold given the l2 norm
        between their prediction and groundtruth.
        Note:
            l2 (np.ndarray): Dimensions are (N x number of keypoints),
            where N is number of frames.
        """
        pck_per_frame_by_thresholds = {}
        for pck_th in thresholds:
            pck_per_frame_by_thresholds[pck_th] = np.nanmean(l2 < pck_th, axis=1)

        pcf_by_thresholds = {}
        for th in thresholds:
            pcf_by_thresholds[th] = np.nanmean(
                pck_per_frame_by_thresholds[th] == 1.0)
        return pcf_by_thresholds

    def _compute_arr_stat(
        self,
        arr: np.ndarray,
        normalization_factor: np.float = 1.0
    ) -> Tuple[np.float, np.float, np.float]:
        """
        Computes statistics for `arr`, and normalizes by the `normalization_factor`.
        Note:
            arr (np.ndarray): Dimensions are (N,).
        """
        arr /= normalization_factor
        return np.nanmean(arr), np.nanstd(arr), \
            np.nanmax(arr) if arr.size else np.nan

    def _compute_pck_per_keypoint(
        self, l2: np.ndarray,
        thresholds: List[int]
    ) -> Dict[int, np.float]:
        """
        Computes Percentage of Correct Keypoints(PCK) per keypoint.
        A keypoint is correct if it is within a threshold given the l2 norm
        between its prediction and groundtruth.
        Note:
            l2 (np.ndarray) : Dimensions are (N x number of keypoints).
        """
        pck_by_thresholds_per_kp = {}
        for pck_th in thresholds:
            pck_by_thresholds_per_kp[pck_th] = np.nanmean(l2 < pck_th, axis=0)
        return pck_by_thresholds_per_kp

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._gt_kps = comm.gather(self._gt_kps, dst=0)
            self._gt_kps = list(itertools.chain(*self._gt_kps))

            self._pred_kps = comm.gather(self._pred_kps, dst=0)
            self._pred_kps = list(itertools.chain(*self._pred_kps))

            self._normalization_factors = comm.gather(
                self._normalization_factors, dst=0)
            self._normalization_factors = list(
                itertools.chain(*self._normalization_factors))

            if not comm.is_main_process():
                return

        l2 = []
        l2_normalized = []
        l1 = []
        l1_normalized = []
        invalid_kp_entries = []
        l2_normalized_temp = []
        l1_normalized_temp = []
        prev_gt_keypoints = None
        prev_pred_keypoints = None

        tic = time.time()
        for i, (normalization_factor, gt_keypoints, pred_keypoints) in enumerate(zip(
            self._normalization_factors, self._gt_kps, self._pred_kps
        )):
            # invalid entries are those with the 'visibility' entry != 2.
            # http://cocodataset.org/#format-data
            invalid_kp_indices = np.argwhere(gt_keypoints[:, 2] != 2)
            invalid_kp_indices = invalid_kp_indices.reshape(
                invalid_kp_indices.shape[0],)

            for kp in invalid_kp_indices:
                invalid_kp_entries.append([i, kp])
            gt_keypoints = gt_keypoints[:, :2]
            pred_keypoints = pred_keypoints[:, :2]

            l2_norm = np.linalg.norm(gt_keypoints - pred_keypoints, axis=1)
            l2_norm_normalized = l2_norm / normalization_factor
            l2.append(l2_norm)
            l2_normalized.append(l2_norm_normalized)

            l1_norm = np.linalg.norm(gt_keypoints - pred_keypoints, ord=1, axis=1)
            l1_norm_normalized = l1_norm / normalization_factor
            l1.append(l1_norm)
            l1_normalized.append(l1_norm_normalized)

            if prev_gt_keypoints is not None and invalid_kp_indices.size == 0:
                gt_keypoints_temp = gt_keypoints - prev_gt_keypoints
                pred_keypoints_temp = pred_keypoints - prev_pred_keypoints

                l2_norm_temp = np.average(np.linalg.norm(gt_keypoints_temp - pred_keypoints_temp, axis=1))
                l2_norm_normalized_temp = l2_norm_temp / normalization_factor
                l2_normalized_temp.append(l2_norm_normalized_temp)

                l1_norm_temp = np.average(np.linalg.norm(gt_keypoints_temp - pred_keypoints_temp, ord=1, axis=1))
                l1_norm_normalized_temp = l1_norm_temp / normalization_factor
                l1_normalized_temp.append(l1_norm_normalized_temp)

                prev_gt_keypoints = gt_keypoints
                prev_pred_keypoints = pred_keypoints
            else:
                l2_normalized_temp.append(np.nan)
                l1_normalized_temp.append(np.nan)

                if invalid_kp_indices.size > 0:
                    prev_gt_keypoints = None
                    prev_pred_keypoints = None
                else:
                    prev_gt_keypoints = gt_keypoints
                    prev_pred_keypoints = pred_keypoints

        l2 = np.array(l2)
        l2_normalized = np.array(l2_normalized)
        l1 = np.array(l1)
        l1_normalized = np.array(l1_normalized)
        l2_normalized_temp = np.array(l2_normalized_temp)
        l1_normalized_temp = np.array(l1_normalized_temp)

        if len(invalid_kp_entries) > 0:
            invalid_kp_entries = np.array(invalid_kp_entries)
            invalid_i, invalid_j = invalid_kp_entries[:, 0], invalid_kp_entries[:, 1]
            l2[invalid_i, invalid_j] = np.nan
            l2_normalized[invalid_i, invalid_j] = np.nan
            l1[invalid_i, invalid_j] = np.nan
            l1_normalized[invalid_i, invalid_j] = np.nan

        L2_mean, L2_std, L2_max = self._compute_arr_stat(l2_normalized)
        L1_mean, L1_std, L1_max = self._compute_arr_stat(l1_normalized)
        L2_mean_jitter, L2_std_jitter, L2_max_jitter = self._compute_arr_stat(l2_normalized_temp)
        L1_mean_jitter, L1_std_jitter, L1_max_jitter = self._compute_arr_stat(l1_normalized_temp)

        pcf_by_thresholds = self._compute_pcf(l2, self._pck_thresholds_in_pixels)

        pck_by_thresholds_per_kp = self._compute_pck_per_keypoint(
            l2, self._pck_thresholds_in_pixels
        )
        logger.info("DONE computing metrics: {:.6}s".format(time.time() - tic))

        metrics = {}
        for pck_threshold in [5, 10, 20]:
            metrics.update({
                "PCK_{}_per_kp_{}".format(
                    pck_threshold,
                    self._metadata.keypoint_names[kp]) : pck
                for kp, pck in enumerate(pck_by_thresholds_per_kp[pck_threshold])
            })
        metrics.update({
            "PCF_per_threshold_{}".format(th) : pcf
            for th, pcf in pcf_by_thresholds.items()
        })
        metrics.update({
            "L1_mean" : L1_mean,
            "L1_std" : L1_std,
            "L1_max" : L1_max,
        })
        metrics.update({
            "L2_mean" : L2_mean,
            "L2_std" : L2_std,
            "L2_max" : L2_max,
        })
        metrics.update({
            "Jitter_L1_mean" : L1_mean_jitter,
            "Jitter_L1_std" : L1_std_jitter,
            "Jitter_L1_max" : L1_max_jitter,
        })
        metrics.update({
            "Jitter_L2_mean" : L2_mean_jitter,
            "Jitter_L2_std" : L2_std_jitter,
            "Jitter_L2_max" : L2_max_jitter,
        })

        output_metrics = OrderedDict({
            "oculus_face" : metrics
        })
        logger.info("==================== Results ====================")
        logger.info("L1 normalized")
        logger.info("  Mean: {}".format(L1_mean))
        logger.info("  Max: {}".format(L1_max))

        logger.info("L2 normalized")
        logger.info("  Mean: {}".format(L2_mean))
        logger.info("  Max: {}".format(L2_max))

        logger.info("Jitter L1 normalized")
        logger.info("  Mean: {}".format(L1_mean_jitter))
        logger.info("  Max: {}".format(L1_max_jitter))

        logger.info("Jitter L2 normalized")
        logger.info("  Mean: {}".format(L2_mean_jitter))
        logger.info("  Max: {}".format(L2_max_jitter))
        logger.info("=================================================")

        return output_metrics
