import torch.nn as nn
from .bevformer.bev_refiner import Bev_refiner
from .bevformer.transformer_decoder import MyTransformeDecoder
import numpy as np

class Traj_refiner(nn.Module):
    def __init__(self,config,init_p=False):
        super().__init__()

        self.poses_num=config.num_poses
        self.state_size=3
        output_dim=self.poses_num*self.state_size

        self.traj_bev = config.traj_bev
        self.b2d = config.b2d


        if init_p:
            input_dim=config.proposal_num
        else:
            input_dim=self.poses_num*self.state_size
            if self.traj_bev:
                self.Bev_refiner=Bev_refiner(config,config.proposal_num,self.poses_num,config.traj_proposal_query)


        self.traj_decoder=MyTransformeDecoder(config,input_dim,output_dim,trajenc=not init_p)

    def forward(self, keyval,proposal_list=None,image_feature=None):

        if len(proposal_list):
            proposals=proposal_list[-1]
            trajectory =proposals.reshape(proposals.shape[0],proposals.shape[1],-1)
            if self.traj_bev:
                keyval = self.Bev_refiner(proposals,keyval,image_feature)
        else:
            trajectory=None

        proposals=self.traj_decoder(trajectory,keyval).reshape(keyval.shape[0],-1,self.poses_num,self.state_size)

        if self.b2d:
            proposals[...,2]+=np.pi/2

        proposal_list.append(proposals)

        return keyval,proposal_list


