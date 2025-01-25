from typing import Any, List, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from pathlib import Path
import pickle
from navsim.agents.pad.pad_model import PadModel
from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
from navsim.planning.training.dataset import load_feature_target_from_pickle
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from navsim.agents.transfuser.transfuser_loss import _agent_loss
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.pad.pad_features import PadTargetBuilder
from navsim.agents.pad.pad_features import PadFeatureBuilder

class PadAgent(AbstractAgent):
    def __init__(
            self,
            config,
            lr: float,
            checkpoint_path: str = None,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path

        cache_data=False

        if not cache_data:
            self._pad_model = PadModel(config)

        if not cache_data and self._checkpoint_path == "":#only for training

            self.bce_logit_loss_nomean = nn.BCEWithLogitsLoss(reduction='none')
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.bce_loss = nn.BCELoss()
            self.ce_loss = nn.CrossEntropyLoss()

            self.agent_loss=_agent_loss

            self.worker = RayDistributedNoTorch()
            self.worker_map=worker_map
            self.b2d = config.b2d

            if config.b2d:
                self.train_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz")
                from .score_module.compute_b2d_score import get_sub_score
                self.get_sub_score = get_sub_score

                map_file ="Bench2DriveZoo/data/infos/b2d_map_infos.pkl"

                with open(map_file, 'rb') as f:
                    map_infos = pickle.load(f)

                self.map_infos = {}
                for town_name, value in map_infos.items():
                    self.map_infos[town_name] = np.concatenate(value['lane_sample_points'], axis=0)[:, :2]

            else:
                from .score_module.compute_sub_score import get_sub_score

                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_sub_score = get_sub_score

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""

        if self._checkpoint_path != "":
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"]
            self.load_state_dict({k.replace("bevrefiner", "pad").replace("agent._pad_model", "_pad_model"): v for k, v in state_dict.items()})

    def get_sensor_config(self) :
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[3],
            cam_l2=[],
            cam_r0=[],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[3],
            lidar_pc=[],
        )
    
    def get_target_builders(self) :
        return [PadTargetBuilder(config=self._config)]

    def get_feature_builders(self) :
        return [PadFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._pad_model(features)

    def compute_score(self, targets, proposals, test=True):

        target_trajectory = targets["trajectory"]

        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        trajectory = proposals.detach().cpu().numpy()

        target_traj = target_trajectory.detach().cpu().numpy()

        if self.b2d:
            data_points = []

            for token, town_name, lidar2world, poses, target_poses in zip(targets["token"], targets["town_name"],
                                                                 targets["lidar2world"].cpu().numpy(), trajectory,
                                                                 target_traj):
                all_lane_points = self.map_infos[town_name]

                xy=lidar2world[0:2, 3]

                dist_to_cur = np.linalg.norm(all_lane_points - xy, axis=-1)

                nearby_point = all_lane_points[dist_to_cur < 50]

                data_dict = {
                    "token": metric_cache_paths[token],
                    "target_trajectory": target_poses,
                    "poses": poses,
                    "lidar2world": lidar2world,
                    "nearby_point": nearby_point
                }
                data_points.append(data_dict)
        else:
            data_points = [
                {
                    "token": metric_cache_paths[token],
                    "poses": poses,
                    "test": test
                }
                for token, poses in zip(targets["token"], trajectory)
            ]

        #all_res = self.get_sub_score(data_points)

        all_res = self.worker_map(self.worker, self.get_sub_score, data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores[:, 1:], l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def score_loss(self, pred_logit, pred_logit2,agents_state, pred_area_logits, target_scores, gt_states, gt_valid,
                   gt_ego_areas,config):

        if agents_state is not None:
            pred_states = agents_state[..., :-gt_states.shape[-3]].reshape(gt_states.shape)
            pred_logits = agents_state[..., -gt_states.shape[-3]:].reshape(gt_valid.shape)

            l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(l1_loss):
                l1_loss = l1_loss.mean()
            else:
                l1_loss = pred_states.mean() * 0

            ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            ce_loss = 0
            l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            ce_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            ce_area_loss = 0

        score_ce_loss = self.bce_logit_loss(pred_logit[..., -6:-1], target_scores[..., -6:-1])  # .mean()

        score_ce_loss_final = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            score_ce_loss2 = self.bce_logit_loss(pred_logit2[..., -6:-1], target_scores[..., -6:-1])  # .mean()

            score_ce_loss_final2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            score_ce_loss=(score_ce_loss+score_ce_loss2)/2

            score_ce_loss_final=(score_ce_loss_final+score_ce_loss_final2)/2

        score_loss = score_ce_loss + score_ce_loss_final + ce_loss + 0.1 * l1_loss + 2 * ce_area_loss  # +sim_loss#+score_real_loss #+best_loss/32

        return score_loss, score_ce_loss, score_ce_loss_final, ce_loss, 0.1 * l1_loss, ce_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        inter_loss = -dist.amin(1).amin(1).mean()

        return inter_loss

    def pad_loss(self,targets: Dict[str, torch.Tensor], pred: Dict[str, torch.Tensor], config  ):

        proposals = pred["proposals"]
        proposal_list = pred["proposal_list"]
        target_trajectory = targets["trajectory"]

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
            targets, proposals, test=False)

        trajectory_loss = 0

        min_loss_list = []
        inter_loss_list = []

        for proposals_i in proposal_list:

            min_loss = torch.linalg.norm(proposals_i - target_trajectory[:, None], dim=-1, ord=1).mean(-1).amin(
                1).mean()

            inter_loss = self.diversity_loss(proposals_i)

            trajectory_loss = 0.1 * trajectory_loss  + min_loss+ inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]

        if "pred_logit" in pred.keys():
            score_loss, score_ce_loss, score_ce_loss1, ce_loss, l1_loss, ce_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas,config)
        else:
            score_loss = score_ce_loss = score_ce_loss1 = ce_loss = l1_loss = ce_area_loss = 0

        if pred["agent_states"] is not None:
            agent_class_loss, agent_box_loss = self.agent_loss(targets, pred, config)
        else:
            agent_class_loss = 0
            agent_box_loss = 0

        if pred["bev_semantic_map"] is not None:
            bev_semantic_loss = F.cross_entropy(pred["bev_semantic_map"], targets["bev_semantic_map"].long())
        else:
            bev_semantic_loss = 0

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.agent_class_weight * agent_class_loss
                + config.agent_box_weight * agent_box_loss
                + config.bev_semantic_weight * bev_semantic_loss
                + config.score_weight * score_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            "agent_class_loss": agent_class_loss,
            "agent_box_loss": agent_box_loss,
            "bev_semantic_loss": bev_semantic_loss,
            "score_loss": score_loss,
            "score": score,
            "best_score": best_score,
            'score_ce_loss': score_ce_loss,
            'score_ce_loss1': score_ce_loss1,
            'ce_loss': ce_loss,
            'l1_loss': l1_loss,
            'ce_area_loss': ce_area_loss,
            "inter_loss0": inter_loss0,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            "min_loss": min_loss,
        }

        return loss_dict

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.pad_loss(targets, pred, self._config)

    def get_optimizers(self):
        return torch.optim.Adam(self._pad_model.parameters(), lr=self._lr)

    def get_training_callbacks(self):

        checkpoint_cb = ModelCheckpoint(save_top_k=100,
                                        monitor='val/score_epoch',
                                        filename='{epoch}-{step}',
                                        )

        return [checkpoint_cb]