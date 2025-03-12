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
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.agents.pad.pad_features import PadTargetBuilder
from navsim.agents.pad.pad_features import PadFeatureBuilder
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from .score_module.compute_b2d_score import compute_corners_torch

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
            self.bce_logit_loss = nn.BCEWithLogitsLoss()
            self.b2d = config.b2d

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=16)
                self.worker_map=worker_map

            if config.b2d:
                self.train_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/train_fut_boxes.gz")
                self.test_metric_cache_paths = load_feature_target_from_pickle(
                    os.getenv("NAVSIM_EXP_ROOT") + "/B2d_cache/val_fut_boxes.gz")
                from .score_module.compute_b2d_score import get_scores
                self.get_scores = get_scores

                map_file =os.getenv("NAVSIM_EXP_ROOT") +"/map.pkl"

                with open(map_file, 'rb') as f:
                    self.map_infos = pickle.load(f)
                self.cuda_map=False

            else:
                from .score_module.compute_navsim_score import get_scores

                metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/train_metric_cache"))
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths

                self.get_scores = get_scores

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


        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        target_trajectory = targets["trajectory"].detach()
        proposals=proposals.detach()

        trajectory = proposals.cpu().numpy()
        target_traj = target_trajectory.cpu().numpy()

        if self.b2d:
            data_points = []

            lidar2worlds=targets["lidar2world"]

            all_proposals = torch.cat([proposals, target_trajectory[:,None]], dim=1)

            all_proposals_xy=all_proposals[:, :,:, :2]
            all_proposals_heading=all_proposals[:, :,:, 2:]

            all_pos = all_proposals_xy.reshape(len(target_traj),-1, 2)

            mid_points = (all_pos.amax(1) + all_pos.amin(1)) / 2

            dists = torch.linalg.norm(all_pos - mid_points[:,None], dim=-1).amax(1) + 5

            xyz = torch.cat(
                [mid_points[..., :2], torch.zeros_like(mid_points[..., :1]), torch.ones_like(mid_points[..., :1])], dim=-1)

            xys = torch.einsum("nij,nj->ni", lidar2worlds, xyz)[:, :2]

            vel = all_proposals_xy[:,:, 1:] - all_proposals_xy[:,:, :-1]

            vel=torch.cat([all_proposals_xy[:, :,:1],vel],dim=2)/ 0.5

            proposals_05 = torch.cat([all_proposals_xy + vel*0.5, all_proposals_heading], dim=-1)

            proposals_ttc = torch.stack([all_proposals, proposals_05], dim=3)

            ego_corners_ttc = compute_corners_torch(proposals_ttc.reshape(-1, 3)).reshape(proposals_ttc.shape[0],proposals_ttc.shape[1], proposals_ttc.shape[2],2,  4, 2)

            ego_corners_center = torch.cat([ego_corners_ttc[:,:,:,0], all_proposals_xy[:, :, :, None]], dim=-2)

            ego_corners_center_xyz = torch.cat(
                [ego_corners_center, torch.zeros_like(ego_corners_center[..., :1]), torch.ones_like(ego_corners_center[..., :1])], dim=-1)

            global_ego_corners_centers = torch.einsum("nij,nptkj->nptki", lidar2worlds, ego_corners_center_xyz)[..., :2]

            l2 = torch.linalg.norm(proposals[..., :2] - target_trajectory[:,None, ..., :2], dim=-1).mean(-1)

            min_indexs = torch.argmin(l2, dim=1)

            vel=vel[:,:-1]

            accs = torch.linalg.norm(vel[:,:, 1:] - vel[:,:, :-1], dim=-1) / 0.5

            comforts = (accs < 10).all(-1)
            
            if self.cuda_map==False:
                for key, value in self.map_infos.items():
                    self.map_infos[key] = torch.tensor(value).to(target_trajectory.device)
                self.cuda_map=True

            for token, town_name, min_index, comfort, dist, xy,global_conners,_ego_coords in zip(targets["token"], targets["town_name"],  min_indexs.cpu().numpy(), comforts.cpu().numpy(), dists.cpu().numpy(), xys, global_ego_corners_centers,ego_corners_ttc.cpu().numpy()):
                all_lane_points = self.map_infos[town_name[:6]]

                dist_to_cur = torch.linalg.norm(all_lane_points[:,:2] - xy, dim=-1)

                nearby_point = all_lane_points[dist_to_cur < dist]

                lane_xy = nearby_point[:, :2]
                lane_width = nearby_point[:, 2]
                lane_id = nearby_point[:, -1]

                dist_to_lane = torch.linalg.norm(global_conners[None] - lane_xy[:, None, None, None], dim=-1)

                on_road = dist_to_lane[:, :-1] < lane_width[:, None, None, None]

                on_road_all = on_road.any(0).all(-1)

                nearest_lane = torch.argmin(dist_to_lane - lane_width[:, None, None,None], dim=0)

                nearest_lane_id=lane_id[nearest_lane]

                center_nearest_lane_id=nearest_lane_id[:,:,-1]

                nearest_road_id = torch.round(center_nearest_lane_id)

                target_road_id = torch.unique(nearest_road_id[-1])

                proposal_center_road_id = nearest_road_id[:-1]

                on_route_all = torch.isin(proposal_center_road_id, target_road_id)
                # in_multiple_lanes: if
                # - more than one drivable polygon contains at least one corner
                # - no polygon contains all corners
                corner_nearest_lane_id=nearest_lane_id[:-1,:,:-1]

                batch_multiple_lanes_mask = (corner_nearest_lane_id!=corner_nearest_lane_id[:,:,:1]).any(-1)

                ego_areas=torch.stack([batch_multiple_lanes_mask,on_road_all,on_route_all],dim=-1)

                data_dict = {
                    "fut_box_corners": metric_cache_paths[token],
                    "_ego_coords": _ego_coords[:-1],
                    "min_index": min_index,
                    "comfort": comfort,
                    "ego_areas": ego_areas.cpu().numpy(),
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

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

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

            pred_l1_loss = F.l1_loss(pred_states, gt_states, reduction="none")[gt_valid]

            if len(pred_l1_loss):
                pred_l1_loss = pred_l1_loss.mean()
            else:
                pred_l1_loss = pred_states.mean() * 0

            pred_ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_valid.to(torch.float32), reduction="mean")

        else:
            pred_ce_loss = 0
            pred_l1_loss = 0

        if pred_area_logits is not None:
            pred_area_logits = pred_area_logits.reshape(gt_ego_areas.shape)

            pred_area_loss = F.binary_cross_entropy_with_logits(pred_area_logits, gt_ego_areas.to(torch.float32),
                                                              reduction="mean")
        else:
            pred_area_loss = 0

        sub_score_loss = self.bce_logit_loss(pred_logit[..., -6:-1], target_scores[..., -6:-1])  # .mean()

        final_score_loss = self.bce_logit_loss(pred_logit[..., -1], target_scores[..., -1])  # .mean()

        if pred_logit2 is not None:
            sub_score_loss2 = self.bce_logit_loss(pred_logit2[..., -6:-1], target_scores[..., -6:-1])  # .mean()

            final_score_loss2 = self.bce_logit_loss(pred_logit2[..., -1], target_scores[..., -1])  # .mean()

            sub_score_loss=(sub_score_loss+sub_score_loss2)/2

            final_score_loss=(final_score_loss+final_score_loss2)/2

        return sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss

    def diversity_loss(self, proposals):
        dist = torch.linalg.norm(proposals[:, :, None] - proposals[:, None], dim=-1, ord=1).mean(-1)

        dist = dist + (dist == 0)

        #dist[dist==0]=10000

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

            trajectory_loss = config.prev_weight * trajectory_loss  + min_loss+ inter_loss * config.inter_weight

            min_loss_list.append(min_loss)
            inter_loss_list.append(inter_loss)

        min_loss0 = min_loss_list[0]
        inter_loss0 = inter_loss_list[0]

        if "pred_logit" in pred.keys():
            sub_score_loss, final_score_loss, pred_ce_loss, pred_l1_loss, pred_area_loss = self.score_loss(
                pred["pred_logit"],pred["pred_logit2"],
                pred["pred_agents_states"], pred["pred_area_logit"]
                , target_scores, gt_states, gt_valid, gt_ego_areas,config)
        else:
            sub_score_loss = final_score_loss = pred_ce_loss = pred_l1_loss = pred_area_loss = 0

        loss = (
                config.trajectory_weight * trajectory_loss
                + config.sub_score_weight * sub_score_loss
                + config.final_score_weight * final_score_loss
                + config.pred_ce_weight * pred_ce_loss
                + config.pred_l1_weight * pred_l1_loss
                + config.pred_area_weight * pred_area_loss
        )

        pdm_score = pred["pdm_score"].detach()
        top_proposals = torch.argmax(pdm_score, dim=1)
        score = final_scores[np.arange(len(final_scores)), top_proposals].mean()
        best_score = best_scores.mean()

        loss_dict = {
            "loss": loss,
            "trajectory_loss": trajectory_loss,
            'sub_score_loss': sub_score_loss,
            'final_score_loss': final_score_loss,
            'pred_ce_loss': pred_ce_loss,
            'pred_l1_loss': pred_l1_loss,
            'pred_area_loss': pred_area_loss,
            "inter_loss0": inter_loss0,
            "inter_loss": inter_loss,
            "min_loss0": min_loss0,
            "min_loss": min_loss,
            "score": score,
            "best_score": best_score
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
        # optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)#,weight_decay= 1e-2
        # self.lr_warmup_steps=500
        # self.lr_total_steps=20000
        # self.lr_min_ratio=1e-3

        # def lr_lambda(current_step):
        #     if current_step < self.lr_warmup_steps:
        #         return (1.0 / 3) + (current_step / self.lr_warmup_steps) * (1 - 1.0 / 3)
        #     return 1.0
        #     # return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
        #     #     1.0
        #     #     + math.cos(
        #     #         math.pi
        #     #         * min(
        #     #             1.0,
        #     #             (current_step - self.lr_warmup_steps)
        #     #             / (self.lr_total_steps - self.lr_warmup_steps),
        #     #         )
        #     #     )
        #     # )


        # scheduler = {
        #     'scheduler': LambdaLR(optimizer, lr_lambda),
        #     'interval': 'step',  # Update every step
        #     'frequency': 1
        # }

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return torch.optim.Adam(self._pad_model.parameters(), lr=self._lr)#,weight_decay=1e-4

    def get_training_callbacks(self):

        checkpoint_cb = ModelCheckpoint(save_top_k=100,
                                        monitor='val/score_epoch',
                                        filename='{epoch}-{step}',
                                        )

        return [checkpoint_cb]