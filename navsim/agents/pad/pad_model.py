from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import Scorer
from .traj_refiner import Traj_refiner
from .bevformer.image_encoder import ImgEncoder

class PadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._config = config

        self._backbone = ImgEncoder(config)

        self.command_num=config.command_num

        self.hist_encoding = nn.Linear(11, config.tf_d_model)

        traj_refiner0 = Traj_refiner(config,init_p=True)

        ref_num=config.ref_num

        self._trajectory_head=nn.ModuleList([traj_refiner0]+[Traj_refiner(config) for _ in range(ref_num) ] )

        self.batch_score = False
        self.scorer = Scorer(config)

        self.b2d=config.b2d

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ego_status: torch.Tensor = features["ego_status"]
        camera_feature: torch.Tensor = features["camera_feature"]

        batch_size = ego_status.shape[0]

        image_feature = self._backbone(camera_feature,img_metas=features)  # b,64,64,64

        output={}

        cur_state=ego_status[:,-1]

        keyval=self.hist_encoding(cur_state)[:,None]

        proposal_list = []
        for i, refine in enumerate(self._trajectory_head):
            keyval, proposal_list = refine(keyval, proposal_list, image_feature)

        proposals=proposal_list[-1]

        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        p_size=proposals.shape[1]

        if self.batch_score:
            keyval=keyval.repeat_interleave(p_size,0)
            image_feature=(image_feature[0].repeat_interleave(p_size,2),
                           image_feature[1],image_feature[2],
                           image_feature[3]['img_metas']['lidar2img'].repeat_interleave(p_size, 0)
                           )
            proposals = proposals.reshape(batch_size*p_size,1,8,-1).detach()[...,:3]
        else:
            proposals = proposals.detach()[...,:3]

        pred_logit,pred_logit2, pred_agents_states, pred_area_logit,bev_semantic_map,agent_states,agent_labels = self.scorer(proposals, keyval,image_feature)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] =agent_labels

        if pred_logit2 is not None:
            pdm_score=(torch.sigmoid(pred_logit)+torch.sigmoid(pred_logit2))[:,:,-1]/2
        else:
            pdm_score=torch.sigmoid(pred_logit)[:,:,-1]

        # if self.b2d:
        #     vel=(proposals[:,:,1:,:2]-proposals[:,:,:-1,:2])/0.5

        #     current_speed=cur_state[:,0]

        #     target=cur_state[:,2:4]

        #     desired_speed=torch.linalg.norm(vel,dim=-1).mean(-1)

        #     acc=(desired_speed-current_speed[:,None])/0.5

        #     distance = torch.linalg.norm((proposals[:,:,1:,:2]+proposals[:,:,:-1,:2])/ 2.0,dim=-1)

        #     gap = torch.abs(distance-4).reshape(-1,distance.shape[-1])

        #     best_gap=torch.argmin(gap,dim=-1)

        #     best_aim=proposals.reshape(best_gap.shape[0],-1,3)[torch.arange(len(best_gap)),best_gap[None]].reshape(proposals.shape[0],proposals.shape[1],3)

        #     angle=torch.arctan2(best_aim[:,:,1], best_aim[:,:,0])

        #     angle_last = torch.arctan2(vel[:,:,-1,1], vel[:,:,-1,0])

        #     angle_target= torch.arctan2(target[:,1], target[:,0])[:,None]

        #     use_target_to_aim = torch.abs(angle_target) < torch.abs(angle)
        #     use_target_to_aim2 = (torch.abs(angle_target-angle_last) > (0.3*2/np.pi)) & (target[:,1] < 10)[:,None]
            
        #     angle[use_target_to_aim]=angle_target
        #     angle[use_target_to_aim2]=angle_target

        #     angel_diff=angle-np.pi/2

        #     lat_acc=acc*torch.sin(angel_diff)
        #     lon_acc=acc*torch.cos(angel_diff)

        #     comfort=(torch.abs(lat_acc)<4.89 ) & (lon_acc<2.40 ) & (lon_acc>-4.05) & (torch.abs(angel_diff*2)<0.95 ) & (desired_speed<16)

        #     pdm_score+=0.2*comfort

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(batch_size), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output



