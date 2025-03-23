from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder
from navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
from navsim.agents.transfuser.transfuser_model import AgentHead,TrajectoryHead

class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self.transfuser_backbone=True

        if self.transfuser_backbone:
            self._backbone= TransfuserBackbone(config)
            self._keyval_embedding = nn.Embedding(8 ** 2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
            self._query_splits = [
                1,
                config.num_bounding_boxes,
            ]

            self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

            # usually, the BEV features are variable in size.
            self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)

            self._bev_semantic_head = nn.Sequential(
                nn.Conv2d(
                    config.bev_features_channels,
                    config.bev_features_channels,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=(1, 1),
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    config.bev_features_channels,
                    config.num_bev_classes,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=True,
                ),
                nn.Upsample(
                    size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                    mode="bilinear",
                    align_corners=False,
                ),
            )

            tf_decoder_layer = nn.TransformerDecoderLayer(
                d_model=config.tf_d_model,
                nhead=config.tf_num_head,
                dim_feedforward=config.tf_d_ffn,
                dropout=config.tf_dropout,
                batch_first=True,
            )

            self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
            self._agent_head = AgentHead(
                num_agents=config.num_bounding_boxes,
                d_ffn=config.tf_d_ffn,
                d_model=config.tf_d_model,
            )
        else:
            self._backbone = ImgEncoder(config)


        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)

        traj_refiner0 = Traj_refiner(config,init_p=True)

        ref_num=config.ref_num

        self._trajectory_head=nn.ModuleList([traj_refiner0]+[Traj_refiner(config) for _ in range(ref_num) ] )

        self.scorer = Scorer(config)

        self.b2d=config.b2d

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"][:,-1]
        camera_feature: torch.Tensor = features["camera_feature"]

        batch_size = ego_status.shape[0]

        if self.b2d:
            
            # if self.training:
            #     ego_status[:,:5]+=torch.randn_like(ego_status[:,:5])

            # print(ego_status[:,1:3])
            # print(ego_status[:,:1])

            ego_status[:,1:3]=0#torch.clamp(ego_status[:,1:3], min=-1, max=1)

            # ego_status=torch.clamp(ego_status, min=-15, max=15)
            ego_status[:,:1]=torch.clamp(ego_status[:,:1], min=0, max=15)

            # ego_status[:,1:3]+=torch.randn_like(ego_status[:,1:3])
            # ego_status[:,:1]+=torch.randn_like(ego_status[:,:1])

        keyval=self.hist_encoding(ego_status)[:,None]
        output={}

        if self.transfuser_backbone:
            lidar_feature: torch.Tensor = features["lidar_feature"]

            bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
            bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
            bev_feature = bev_feature.permute(0, 2, 1)

            keyval = torch.concatenate([bev_feature, keyval], dim=1)
            keyval += self._keyval_embedding.weight[None, ...]
            query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
            query_out = self._tf_decoder(query, keyval)

            bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
            trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

            output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

            agents = self._agent_head(agents_query)
            output.update(agents)
            image_feature=None

        else:
            image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64


        proposal_list = []
        for i, refine in enumerate(self._trajectory_head):
            keyval, proposal_list = refine(keyval, proposal_list, image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        proposals = proposals.detach()[..., :3]

        pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels = self.scorer(proposals, keyval,image_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit

        if "bev_semantic_map" not in output.keys():
            output["bev_semantic_map"]=bev_semantic_map
            output["agent_states"] = agent_states
            output["agent_labels"] =agent_labels

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output



