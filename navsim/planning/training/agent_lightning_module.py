import numpy as np
import pytorch_lightning as pl
import torch

from torch import Tensor
from typing import Dict, Tuple, Any

from navsim.agents.abstract_agent import AbstractAgent
import pickle
import torch.distributed as dist

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        #features["trajectory"]=targets["trajectory"]
        # features["current_epoch"]=self.trainer.current_epoch
        # targets["current_epoch"]=self.trainer.current_epoch

        prediction = self.agent.forward(features)
        loss_dict = self.agent.compute_loss(features, targets, prediction)

        if type(loss_dict) is dict:
            for key,value in loss_dict.items():
                self.log(f"{logging_prefix}/"+key, value, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

            return loss_dict["loss"]
        else:
            return loss_dict


    # def on_train_epoch_start(self) -> None:
    #     # if self.trainer.current_epoch==self.agent.learn_epoch:
    #     #     self.trainer.optimizers=[torch.optim.Adam(self.agent._pad_model.parameters(), lr=0.0001)]
    #
    #     if self.agent.opt_traj:
    #
    #         gather_scores = [torch.zeros_like(self.agent.best_score)  for dev_idx in range(torch.cuda.device_count())]
    #
    #         dist.all_gather(gather_scores, self.agent.best_score)
    #
    #         gather_trajs = [torch.zeros_like(self.agent.best_traj) for dev_idx in range(torch.cuda.device_count())]
    #         dist.all_gather(gather_trajs, self.agent.best_traj)
    #
    #         for gather_score,gather_traj in zip(gather_scores,gather_trajs):
    #             improve=gather_score>self.agent.best_score
    #             self.agent.best_score[improve] = gather_score[improve]
    #             self.agent.best_traj[improve]=gather_traj[improve]

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
        if 'Bev' in self.agent.name():
            features, targets = batch
            # score,best_score=self.agent.inference(features, targets)
            predictions = self.agent.forward(features)
            all_res=torch.cat([predictions["trajectory"][:,None],predictions["proposals"]],dim=1)
            final_score,best_score,proposal_scores,l2,trajectoy_scores=self.agent.compute_score(targets,all_res)
            mean_score=proposal_scores.mean()
            pdm_score=predictions["pdm_score"]
            score_error=torch.abs(pdm_score - proposal_scores).mean()
            logging_prefix="val"
            self.log(f"{logging_prefix}/score", final_score, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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



    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
