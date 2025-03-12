from time import sleep

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Tuple, Any

from navsim.agents.abstract_agent import AbstractAgent
import glob
import os
import subprocess
import shutil
import json

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
        self.checkpoint_file=None

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
            return loss_dict["loss"]
        else:
            return loss_dict

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        if 'Pad' in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_res=torch.cat([predictions["trajectory"][:,None],predictions["proposals"]],dim=1)
            final_score,best_score,proposal_scores,l2,trajectoy_scores=self.agent.compute_score(targets,all_res)
            mean_score=proposal_scores.mean()
            pdm_score=predictions["pdm_score"]
            score_error=torch.abs(pdm_score - proposal_scores).mean()
            logging_prefix="val"
            self.log(f"{logging_prefix}/score", final_score, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/best_score", best_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/mean_score", mean_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/score_error", score_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            self.log(f"{logging_prefix}/l2", l2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            collision=trajectoy_scores[:,0].mean()
            self.log(f"{logging_prefix}/collision", collision, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            drivable_area_compliance=trajectoy_scores[:,1].mean()
            self.log(f"{logging_prefix}/dac", drivable_area_compliance, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            ego_progress=trajectoy_scores[:,2].mean()
            self.log(f"{logging_prefix}/progress", ego_progress, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            time_to_collision_within_bound=trajectoy_scores[:,3].mean()
            self.log(f"{logging_prefix}/ttc", time_to_collision_within_bound, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            comfort=trajectoy_scores[:,4].mean()
            self.log(f"{logging_prefix}/comfort", comfort, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return final_score
        else:
            return self._step(batch, "val")

    # def on_validation_epoch_end(self) -> None:
    #     if self.agent.b2d and self.checkpoint_file is not None:
    #
    #         folder_path=self.trainer.default_root_dir+'/res_'+self.checkpoint_file[:-5]
    #         subprocess.run(["pkill", "-9", "-f", "leaderboard_evaluator"])
    #         subprocess.run(["pkill", "-9", "-f", "carla"])
    #
    #         file_paths = glob.glob(f'{folder_path}/*.json')
    #         merged_records = []
    #         driving_score = []
    #         success_num = 0
    #         for file_path in file_paths:
    #             if 'merged.json' in file_path: continue
    #             with open(file_path) as file:
    #                 data = json.load(file)
    #                 records = data['_checkpoint']['records']
    #                 for rd in records:
    #                     rd.pop('index')
    #                     merged_records.append(rd)
    #                     driving_score.append(rd['scores']['score_composed'])
    #                     if rd['status'] == 'Completed' or rd['status'] == 'Perfect':
    #                         success_flag = True
    #                         for k, v in rd['infractions'].items():
    #                             if len(v) > 0 and k != 'min_speed_infractions':
    #                                 success_flag = False
    #                                 break
    #                         if success_flag:
    #                             success_num += 1
    #                             print(rd['route_id'])
    #
    #         if len(merged_records):
    #             driving_score=sum(driving_score) / len(merged_records)
    #             success_rate= success_num / len(merged_records)
    #         else:
    #             driving_score=0
    #             success_rate=0
    #
    #         eval_num=len(merged_records)
    #
    #         logging_prefix = "val"
    #
    #         self.log(f"{logging_prefix}/driving_score", driving_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #         self.log(f"{logging_prefix}/success_rate", success_rate, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #         self.log(f"{logging_prefix}/eval_num", eval_num, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #
    # def on_train_epoch_start(self):
    #     if  self.agent.b2d and self.global_step>0:
    #         checkpoint_path=self.trainer.default_root_dir+"/lightning_logs/version_0/checkpoints"
    #         for checkpoint_file in os.listdir(checkpoint_path):
    #             if str(self.global_step) in checkpoint_file:
    #                 self.checkpoint_file=checkpoint_file
    #         checkpoint_path=checkpoint_path+'/'+self.checkpoint_file
    #
    #         result_dir=self.trainer.default_root_dir+'/res_'+self.checkpoint_file[:-5]
    #
    #         if self.global_rank==0:
    #             os.makedirs(result_dir)
    #
    #         closeloop_eval_script='leaderboard/scripts/run_evaluation_pad.sh'
    #
    #         global_rank =self.global_rank  # Replace with your actual global_rank, or use self.global_rank if inside a class
    #
    #         # Construct the command arguments
    #         command = ['bash', closeloop_eval_script, checkpoint_path, str(global_rank),result_dir]
    #
    #         subprocess.run(command, cwd=os.getenv('Bench2Drive_ROOT'))

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
