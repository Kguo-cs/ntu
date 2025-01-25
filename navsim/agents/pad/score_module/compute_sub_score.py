import torch

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
from .train_pdm_scorer import PDMScorerConfig, PDMScorer

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)

def get_sub_score(args):

    return [get_sub_score1(a["token"],a["poses"],a["test"]) for a in args]


def get_sub_score1( metric_cache,poses,test):

    with lzma.open(metric_cache, "rb") as f:
        metric_cache = pickle.load(f)

    initial_ego_state = metric_cache.ego_state

    trajectory_states = []

    for model_trajectory in poses:
        pred_trajectory = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)

        pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling,
                                              initial_ego_state.time_point)

        trajectory_states.append(pred_states)

    trajectory_states = np.stack(trajectory_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)#32,41,11

    final_scores=scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.pdm_progress
    )

    num_col=2

    key_agent_corners = np.zeros([len(final_scores),num_col, scorer.proposal_sampling.num_poses + 1, 4, 2])
    key_agent_labels = np.zeros([len(final_scores),num_col, scorer.proposal_sampling.num_poses + 1],dtype=bool)
    ego_areas = scorer._ego_areas[:,1:]

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    #driving_direction_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]


    scores=np.stack([no_at_fault_collisions,drivable_area_compliance,#driving_direction_compliance,
                     ego_progress,time_to_collision_within_bound,comfort,final_scores
                     ],axis=-1)#[:,None]

    if not test:
        for i in range(len(scores)):
            # proposal_collided_track_ids=scorer.proposal_collided_track_ids[i]
            proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
            # temp_collided_track_ids=scorer.temp_collided_track_ids[i]

            if len(proposal_fault_collided_track_ids):
                col_token=proposal_fault_collided_track_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[i])+1

                for time_idx in range(collision_time_idcs):
                    if  col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][0][time_idx] = True
                        key_agent_corners[i][0][time_idx]=np.array(scorer._observation[time_idx][col_token].boundary.xy).T[:4]

            ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]

            if len(ttc_collided_track_ids):
                ttc_token=ttc_collided_track_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[i])+1

                for time_idx in range(ttc_time_idcs):
                    if  ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][1][time_idx] = True
                        key_agent_corners[i][1][time_idx]=np.array(scorer._observation[time_idx][ttc_token].boundary.xy).T[:4]

        theta = initial_ego_state.rear_axle.heading
        origin_x = initial_ego_state.rear_axle.x
        origin_y = initial_ego_state.rear_axle.y

        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])

        key_agent_corners[...,0]-=origin_x
        key_agent_corners[...,1]-=origin_y

        key_agent_corners=key_agent_corners.dot(mat)

    return scores,key_agent_corners,key_agent_labels,ego_areas
# ego_progress=np.histogram(ego_progress)

# scores=np.stack([drivable_area_compliance,
#                  no_at_fault_collisions==1,
#                  no_at_fault_collisions==0.5,
#                  ego_progress,
#                  time_to_collision_within_bound,
#                  comfort,
#                  ],axis=-1)[:,None]

# print(np.where(no_at_fault_collisions==0.5))

# for j,token in enumerate(proposal_collided_track_ids[:num_col//4]):
#     if  token in scorer._observation[time_idx].tokens:
#         key_agent_labels[i][j][time_idx] = True
#         tracked_object_polygon=scorer._observation[time_idx][token]
#
#         corner_point=np.array(tracked_object_polygon.boundary.xy).T[:4]
#
#         key_agent_corners[i][j][time_idx]=corner_point

# for j,token in enumerate(temp_collided_track_ids[:num_col//4]):
#     if  token in scorer._observation[time_idx].tokens:
#         key_agent_labels[i][j+num_col//4*3][time_idx] = True
#         tracked_object_polygon=scorer._observation[time_idx][token]
#
#         corner_point=np.array(tracked_object_polygon.boundary.xy).T[:4]
#
#         key_agent_corners[i][j+num_col//4*3][time_idx]=corner_point

# print("trajectory_states",poses[0])
#  print("no_at_fault_collisions",no_at_fault_collisions.mean())
#  print("drivable_area_compliance",drivable_area_compliance.mean())
#  print("ego_progress",ego_progress.mean())
#  print("time_to_collision_within_bound",time_to_collision_within_bound.mean())
#  print("comfort",comfort.mean())
#  print(len(metric_cache.observation.unique_objects),len(metric_cache.centerline.discrete_path))
#  print(metric_cache.pdm_progress)
#  print(len(metric_cache.drivable_area_map))
#  print("fina",scores.mean())
#  print("fina",simulated_states[0])
# print("initial_ego_state",initial_ego_state.rear_axle)
# print("initial_ego_state",initial_ego_state.dynamic_car_state)

# tracked_object = scorer._observation.unique_objects[token]
# type=tracked_object.tracked_object_type

# tracked_object = scorer._observation.unique_objects[token]
 # _X = 0
    # _Y = 1
    # _HEADING = 2
    # _VELOCITY_X = 3
    # _VELOCITY_Y = 4
    # _ACCELERATION_X = 5
    # _ACCELERATION_Y = 6
    # _STEERING_ANGLE = 7
    # _STEERING_RATE = 8
    # _ANGULAR_VELOCITY = 9
    # _ANGULAR_ACCELERATION = 10
    #
    #     simulated_traj=simulated_states[:,1:,:2]#-origin_x
    #
    #     simulated_traj[...,0]-=origin_x
    #     simulated_traj[...,1]-=origin_y
    #
    #     simulated_traj=simulated_traj.dot(mat)
    #
    #     simulated_traj=simulated_traj.reshape(-1,40*2)
    #
    #     simulated_vel=simulated_states[:,1:,3:5]#-origin_x
    #     simulated_acc=simulated_states[:,1:,5:7]#-origin_x
    #
    #     simulated_vel=simulated_vel.dot(mat)
    #     simulated_acc=simulated_acc.dot(mat)
    #
    #     simulated_vel=simulated_vel.reshape(-1,40*2)
    #     simulated_acc=simulated_acc.reshape(-1,40*2)
    #
    #     heading=simulated_states[:,1:,2]
    #     steering_angle=simulated_states[:,1:,7]
    #
    #     rel_heading = (heading - theta +  np.pi) % (2*np.pi)-np.pi
    #     rel_steering_angle = (steering_angle - theta +  np.pi) % (2*np.pi)-np.pi
    #
    #     # traj_heading=np.concatenate([simulated_traj,rel_heading[...,None]],axis=-1)
    #     rate=simulated_states[:,1:,-3:].reshape(-1,40*3)
    #     # traj_heading=traj_heading.reshape(traj_heading.shape[0],-1)
    #
    #     # scores=np.concatenate([simulated_traj,rel_heading,simulated_vel,simulated_acc,
    #     #     rel_steering_angle,rate,scores],axis=-1)
    #     scores=np.concatenate([simulated_traj,rel_heading,scores],axis=-1)

        # _collision_time_idcs=scorer._collision_time_idcs[:,None]
        # _ttc_time_idcs=scorer._ttc_time_idcs[:,None]
        # ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :,None]
        #
        # _progress_raw=scorer._progress_raw
        # # # key_agent_corners=key_agent_corners.reshape(len(scores),num_col,-1)
        # #
        # # other_feature=np.stack([_collision_time_idcs,_ttc_time_idcs,_progress_raw],axis=-1)
        #
        # _collision_time_idcs[_collision_time_idcs==np.inf]=100
        # _ttc_time_idcs[_ttc_time_idcs==np.inf]=100
        #
        # driving_direction_compliance=scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION,:,None]
        #
        # score_features=np.concatenate([_collision_time_idcs/100,_ttc_time_idcs/100,driving_direction_compliance ,ego_progress,scorer.off_road,scorer.is_comfortable ],axis=-1)

        #scores=np.concatenate([scores,score_features],axis=-1)