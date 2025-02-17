from dataclasses import dataclass

@dataclass
class PadConfig:
    b2d = True

    ref_num: int=1

    traj_bev: bool=True
    score_bev: bool=True

    traj_proposal_query: bool=True
    score_proposal_query: bool=True

    bev_map: bool=True
    bev_agent: bool=True

    double_score: bool=True
    agent_pred: bool=True
    area_pred: bool=True

    proposal_num: int = 32
    inter_weight: float =0.1

    score_weight: int = 10
    sub_score_weight: int = 1

    point_cloud_range= [-32, -32, -2.0, 32, 32,2.0]

    half_length = 2.44619083405+0.15#2.042
    half_width= 0.91835665702+0.1#0.925
    rear_axle_to_center =0.39

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
    tf_dropout: float = 0.0

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    num_bev_classes = 15
    bev_features_channels: int = 64

    lidar_resolution_width = 256
    lidar_resolution_height = 256

    latent: bool = False

