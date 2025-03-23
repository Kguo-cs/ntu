import torch
import torch.nn as nn
from ..bevformer.bev_refiner import Bev_refiner
from ..bevformer.transformer_decoder import MyTransformeDecoder,MLP
from .map_head import MapHead
import numpy as np

class Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.score_bev = config.score_bev

        self.b2d=config.b2d

        self.proposal_num=config.proposal_num
        self.score_num = 6
        num_poses = config.num_poses

        state_size=3

        if self.score_bev:
            self.Bev_refiner=Bev_refiner(config,self.proposal_num,num_poses,proposal_query=config.score_proposal_query)

        input_dim=state_size* num_poses

        self.score_mlp=True

        if self.score_mlp:
            self.pred_score = MLP(config.tf_d_model,config.tf_d_ffn,self.score_num)
        else:
            self.pred_score = MyTransformeDecoder(config,input_dim,self.score_num )

        self.double_score=config.double_score

        if self.double_score:
            if self.score_mlp:
                self.pred_score2 = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)
            else:
                self.pred_score2 = MyTransformeDecoder(config, input_dim, self.score_num)

        self.agent_pred= config.agent_pred

        num_agent_pose=config.num_agent_pose

        self.agent_mlp=True

        if self.agent_pred:
            if self.agent_mlp:
                self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, num_agent_pose * 9)
                self.pred_ttc_agent = MLP(config.tf_d_model, config.tf_d_ffn, num_agent_pose * 9)
            else:
                self.pred_col_agent = MyTransformeDecoder(config, input_dim, num_agent_pose * 9)
                self.pred_ttc_agent = MyTransformeDecoder(config, input_dim, num_agent_pose * 9)

        self.area_pred=config.area_pred

        if self.area_pred:
            if self.b2d:
                self.pred_area = MLP(config.tf_d_model, config.tf_d_ffn, 2)
            else:
                self.pred_area = MLP(config.tf_d_model, config.tf_d_ffn, 5*3)

        self.bev_map=config.bev_map
        self.bev_agent=config.bev_agent

        if config.bev_agent:
            self._agent_head=MyTransformeDecoder(config,config.num_bounding_boxes,6,trajenc=False)

        if config.bev_map:
            self.map_head=MapHead(config)


    def forward(self, proposals,keyval,image_feature):
        batch_size=len(proposals)
        p_size=proposals.shape[1]
        t_size=proposals.shape[2]

        if self.score_bev:
            keyval = self.Bev_refiner(proposals, keyval, image_feature)

        trajectory = proposals.reshape(batch_size,p_size, -1)

        if self.score_mlp:
            proposal_feature = keyval[:, :p_size * t_size].reshape(batch_size,p_size,t_size,-1).amax(-2)
            pred_logit=self.pred_score(proposal_feature).reshape(batch_size, -1, 6)
        else:
            pred_logit=self.pred_score(trajectory,keyval).reshape(batch_size, -1, 6)

        pred_logit2=pred_agents_states=pred_area_logit=bev_semantic_map=agent_states=agent_labels=None

        if self.double_score:
            if self.score_mlp:
                pred_logit2 = self.pred_score2(proposal_feature).reshape(batch_size, -1, 6)
            else:
                pred_logit2 = self.pred_score2(trajectory, keyval).reshape(batch_size, -1, 6)

        if  self.training:
            if self.bev_map:
                bev_semantic_map = self.map_head(keyval)

            if self.bev_agent:
                agents =self._agent_head(None,keyval)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

            if self.area_pred:
                pred_area_logit = self.pred_area(keyval[:, :p_size * t_size])

            if self.agent_pred:
                if self.agent_mlp:
                    col_agents_state=self.pred_col_agent(proposal_feature)
                    ttc_agents_state=self.pred_ttc_agent(proposal_feature)
                else:
                    col_agents_state = self.pred_col_agent(trajectory, keyval)
                    ttc_agents_state = self.pred_ttc_agent(trajectory, keyval)
                pred_agents_states = torch.stack([ttc_agents_state, col_agents_state], dim=2)

        return pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels
