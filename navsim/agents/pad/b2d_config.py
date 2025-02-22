from dataclasses import dataclass

@dataclass
class PadConfig:
    b2d = True

    ref_num: int=1

    traj_bev: bool=True
    score_bev: bool=True

    traj_proposal_query: bool=True
    score_proposal_query: bool=True

    bev_map: bool=False
    bev_agent: bool=False

    double_score: bool=True
    agent_pred: bool=True
    area_pred: bool=True

    proposal_num: int = 64
    point_cloud_range= [-32, -32, 0.0, 32, 32,4.0]

    half_length = 2.042+0.25
    half_width= 0.925+0.1
    rear_axle_to_center =0.39
    lidar_height=2.05

    num_poses=6
    num_agent_pose=6
    command_num=7

    num_bev_layers: int=2
    num_points_in_pillar: int=4

    image_architecture: str = "resnet34"

    # Transformer
    tf_d_model: int = 512
    tf_d_ffn: int = 2048
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.1

    # loss weights
    trajectory_weight: float = 1.0
    inter_weight: float = 0.1
    sub_score_weight: int = 1
    final_score_weight: int = 1
    pred_ce_weight: int = 1
    pred_l1_weight: int = 1
    pred_area_weight: int = 1
    prev_weight: int = 0.1