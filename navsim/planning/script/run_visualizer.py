from typing import Dict
from pathlib import Path
import logging
import traceback
import pickle
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm
from navsim.common.dataclasses import SensorConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Trajectory, SceneFilter
from navsim.common.dataloader import SceneLoader
import torch
from torch.utils.data import default_collate
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_create_submission_pickle"

import matplotlib.pyplot as plt
import numpy as np
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
import lzma
import pickle
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from matplotlib.patches import Polygon
import cv2
import matplotlib.cm as cm

proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)


def run_test_evaluation(
    agent: AbstractAgent, scene_filter: SceneFilter, data_path: Path, sensor_blobs_path: Path
) -> Dict[str, Trajectory]:
    """
    Function to create the output file for evaluation of an agent on the testserver
    :param agent: Agent object
    :param data_path: pathlib path to navsim logs
    :param sensor_blobs_path: pathlib path to sensor blobs
    :param save_path: pathlib path to folder where scores are stored as .csv
    """
    if agent.requires_scene:
        raise ValueError(
            """
            In evaluation, no access to the annotated scene is provided, but only to the AgentInput.
            Thus, agent.requires_scene has to be False for the agent that is to be evaluated.
            """
        )
    logger.info("Building Agent Input Loader")
    input_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs_path,
        sensor_config=SensorConfig.build_all_sensors(include=[3]),
    )
    agent.initialize()

    output: Dict[str, Trajectory] = {}


    agent.eval()
    agent.cuda()

    for token in tqdm(input_loader, desc="Running evaluation"):
        # try:
        # if token not in ["51e697d3f5255ac3","83212ddd15375812", "fc6dc98b89a95817", "dabc9043dbb9560b", "ce516bdfc6e45d5b"]:
        #     continue
        print(token)

        scene=input_loader.get_scene_from_token(token)
        agent_input = input_loader.get_agent_input_from_token(token)

        #print(agent_input.ego_statuses[-1])

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v for k, v in features.items()}

        feature_list=[features]
        
        features=default_collate(feature_list)

        features={key:value.cuda() for key,value in features.items()}
        # forward pass
        with torch.no_grad():
            predictions = agent.forward(features)
        
        poses = predictions["trajectory"].cpu().numpy()[0,:,:2]
        proposals = predictions["proposals"].cpu().numpy()[0]
        pdm_score = predictions["pdm_score"].cpu().numpy()[0]

        # scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")

        # initial_ego_state = scenario.initial_ego_state

        # trajectory_states=[]

        # for model_trajectory in proposals:
        #     pred_trajectory = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)

        #     pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling,
        #                                         initial_ego_state.time_point)

        #     trajectory_states.append(pred_states)

        # trajectory_states = np.stack(trajectory_states, axis=0)

        # simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11

        # simulated_traj=simulated_states[:,1:,:2]
        # theta = initial_ego_state.rear_axle.heading
        # origin_x = initial_ego_state.rear_axle.x
        # origin_y = initial_ego_state.rear_axle.y

        # c, s = np.cos(theta), np.sin(theta)
        # mat = np.array([[c, -s],
        #                 [s, c]])

        # simulated_traj[...,0]-=origin_x
        # simulated_traj[...,1]-=origin_y
    
        # simulated_traj=simulated_traj.dot(mat)
    
        # simulated_traj=simulated_traj.reshape(-1,40,2)


        # pred_area_logit = torch.sigmoid(predictions["pred_area_logit"]).cpu().numpy()[0]
        pred_agents_states = predictions["pred_agents_states"][0]

        pred_agents_label=torch.sigmoid(pred_agents_states[..., -41:]).cpu().numpy()
        pred_agents_corners=pred_agents_states[..., :-41].cpu().numpy()


        camera=agent_input.cameras[-1].cam_f0
        # image0=cam.image

        
            #     np.maximum(
            # reference_points_cam[..., 2:3], np.ones_like(reference_points_cam[..., 2:3]) * eps)

        # print(reference_points_cam)

        # for i in range(64):
        #     #refer_i=reference_points_cam[i][bev_mask[i]]
        #     plt.plot(reference_points_cam[i,:,0], reference_points_cam[i,:,1], color="red", linewidth=1) 

        from navsim.visualization.plots import plot_bev_frame
        from navsim.visualization.plots import plot_bev_with_agent,configure_bev_ax,configure_ax
        from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
        from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
        from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
        from matplotlib.gridspec import GridSpec

        frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
        # fig, ax = plot_bev_frame(scene, frame_idx)
        # plt.show()

        # agent = ConstantVelocityAgent()
        # fig, ax = plot_bev_with_agent(scene, agent)
        human_trajectory = scene.get_future_trajectory(8)
        #agent_trajectory = agent.compute_trajectory(scene.get_agent_input())

        frame_idx = scene.scene_metadata.num_history_frames - 1
        #fig, (ax0 ,ax)= plt.subplots(2, 1, figsize=(5, 6))
        fig = plt.figure(figsize=(8, 8))
        gs = GridSpec(2, 1, figure=fig, hspace=0, wspace=0,height_ratios=[0.36,0.64])

        # Create axes
        ax0 = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])

        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])

        config=TRAJECTORY_CONFIG["human"]

        human_poses = human_trajectory.poses[:, :2]
        ax.plot(
            human_poses[:, 1],
            human_poses[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"]
        )


        config=TRAJECTORY_CONFIG["agent"]#red

        poses = poses[:, :2]
        ax.plot(
            poses[:, 1],
            poses[:, 0],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"]
        )

        ttc_marking=True
        col_marking=True

        for i in range(64):
            proposal=proposals[i]
            color = cm.Reds(pdm_score[i])  # This returns an RGBA color

            ax.plot(
                proposal[:, 1],
                proposal[:, 0],
                color=color,
                linewidth=1,
                linestyle=config["line_style"],
                marker='.',
                markersize=2,
                # markeredgecolor=config["marker_edge_color"],
                zorder=2
            )

            col_label=pred_agents_label[i][0]

            if col_label.max()>0.5:
                #print(i)
                col_corner=pred_agents_corners[i][0].reshape(41,4,2)

                #print(col_label)
                #print(col_corner)
                color = cm.Blues(pdm_score[i])  # This returns an RGBA color

                #print(col_corner)
                for t in range(1,41,5):
                    if col_label[t]>0.3:
                        col_corner_t=col_corner[t][:,::-1]
                        if col_marking:
                            p = Polygon(col_corner_t,
                                alpha=1,
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2,
                                label='At-fault Collision Prediction'
                                )
                        else:
                            p = Polygon(col_corner_t,
                                        alpha=pdm_score[i],
                                        edgecolor=color,
                                        facecolor = "None",
                                        zorder=2
                                        )

                        ax.add_patch(p)
                        col_marking=False

            ttc_label=pred_agents_label[i][1]
            color = cm.Oranges(pdm_score[i])  # This returns an RGBA color

            if ttc_label.max()>0.8:
                #print(i)
                ttc_corner=pred_agents_corners[i][1].reshape(41,4,2)

                #print(ttc_label)
                #print(col_corner)

                #print(col_corner)
                for t in range(1,41,5):
                    if ttc_label[t]>0.5:
                        ttc_corner_t=ttc_corner[t][:,::-1]
                        if ttc_marking and col_marking==False:
                            p = Polygon(ttc_corner_t,
                                alpha=1,
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2,
                                label='Time-to-collision Prediction'
                                )
                            ttc_marking=False

                        else:
                            p = Polygon(ttc_corner_t,
                                alpha=pdm_score[i],
                                edgecolor=color,
                                facecolor = "None",
                                zorder=2
                                )
                        ax.add_patch(p)

        ax.set_aspect("equal")

        # NOTE: x forward, y sideways
        ax.set_xlim(-36, 36)
        ax.set_ylim(-8, 64)

        # NOTE: left is y positive, right is y negative
        ax.invert_xaxis()
        configure_ax(ax)

        from navsim.visualization.camera import _transform_points_to_image,add_lidar_to_camera_ax,_transform_pcs_to_images


        proposals=np.concatenate([ proposals[..., :2],human_poses[None],poses[None,:,:2]],axis=0)

        proposals=np.concatenate([proposals[..., :2], np.zeros_like(proposals)[..., :1]], -1)
        # frame = scene.frames[frame_idx]

        # add_lidar_to_camera_ax(ax0, camera, frame.lidar)
        lidar_pc=proposals.reshape(-1,3).T#frame.lidar.lidar_pc.copy()#
        image = camera.image.copy()

        image_height, image_width = image.shape[:2]

        pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
            lidar_pc,
            camera.sensor2lidar_rotation,
            camera.sensor2lidar_translation,
            camera.intrinsics,
            img_shape=(image_height, image_width),
        )

        pc_in_cam=pc_in_cam.reshape(-1,8,2)
        pc_in_fov_mask=pc_in_fov_mask.reshape(-1,8)



        config=TRAJECTORY_CONFIG["human"]

        ax0.plot(
            pc_in_cam[-2,:, 0],
            pc_in_cam[-2,:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"],
            label="Human Trajectory"
        )


        config=TRAJECTORY_CONFIG["agent"]#red

        ax0.plot(
            pc_in_cam[-1,:, 0],
            pc_in_cam[-1,:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config["marker"],
            markersize=config["marker_size"],
            markeredgecolor=config["marker_edge_color"],
            zorder=config["zorder"],
            label="Planning"
        )


        for i in range(64):
            points=pc_in_cam[i]#[pc_in_fov_mask[i],None].reshape(-1,2)
            color = cm.Reds(pdm_score[i])  # This returns an RGBA color

            if i==0:
                ax0.plot(
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    linewidth=1,
                    linestyle=config["line_style"],
                    marker='.',
                    markersize=2,
                    zorder=2,
                    label="Proposal"
                )
            else:
                ax0.plot(
                    points[:, 0],
                    points[:, 1],
                    color=color,
                    alpha=pdm_score[i],
                    linewidth=1,
                    linestyle=config["line_style"],
                    marker='.',
                    markersize=2,
                    # markeredgecolor=config["marker_edge_color"],
                    zorder=2
                )

        ax0.axis('off')
        ax0.legend(loc=1)
        ax.legend(loc=1)

        ax0.imshow(image)

        # proposals=np.concatenate([proposals[..., :2], np.zeros_like(proposals)[..., :1]], -1)

        # proposal_points, pc_in_fov = _transform_points_to_image(proposals.reshape(-1, 3), camera.intrinsics)

        # proposal_points=proposal_points.reshape(-1,8,2)
        # pc_in_fov=pc_in_fov.reshape(-1,8)
        # valid_proposal = pc_in_fov.any(-1)

        # box_corners, box_labels = box_corners[valid_proposal], box_labels[valid_proposal]
        # image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)
        # image =camera.image.copy()
        # from PIL import ImageColor

        # color = ImageColor.getcolor("#e15759", "RGB")

        # for i in range(64):
        #     proposal_points_i=proposal_points[i].astype(np.int)
            
        #     for t in range(7):
        #         cv2.line(
        #             image,
        #             (proposal_points_i[t, 0], proposal_points_i[t, 1]),
        #             (proposal_points_i[t+1, 0], proposal_points_i[t+1, 1]),
        #             color,
        #             1,
        #             cv2.LINE_AA,
        #         )

        plt.tight_layout()
        plt.show()

        # from navsim.visualization.plots import plot_cameras_frame

        # fig, ax = plot_cameras_frame(scene, frame_idx)
        # plt.show()
        # from navsim.visualization.plots import plot_cameras_frame_with_annotations

        # fig, ax = plot_cameras_frame_with_annotations(scene, frame_idx)
        # plt.show()
        # from navsim.visualization.plots import plot_cameras_frame_with_lidar

        # fig, ax = plot_cameras_frame_with_lidar(scene, frame_idx)
        # plt.show()
        # from navsim.visualization.plots import configure_bev_ax
        # from navsim.visualization.bev import add_annotations_to_bev_ax, add_lidar_to_bev_ax


        # fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # ax.set_title("Custom plot")

        # add_annotations_to_bev_ax(ax, scene.frames[frame_idx].annotations)
        # add_lidar_to_bev_ax(ax, scene.frames[frame_idx].lidar)

        # # configures frame to BEV view
        # configure_bev_ax(ax)

        # plt.show()

        # plt.imshow(image0)

        # plt.show()


    return output


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for submission creation script.
    :param cfg: omegaconf dictionary
    """
    agent = instantiate(cfg.agent)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    save_path = Path(cfg.output_dir)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)

    output = run_test_evaluation(
        agent=agent,
        scene_filter=scene_filter,
        data_path=data_path,
        sensor_blobs_path=sensor_blobs_path,
    )

if __name__ == "__main__":
    main()
