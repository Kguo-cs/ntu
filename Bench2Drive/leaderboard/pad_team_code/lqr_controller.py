from collections import deque
import numpy as np
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateVector2D, StateSE2, TimePoint
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_kinematic_bicycle import BatchKinematicBicycleModel
from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr import BatchLQRTracker
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration

class LQRController(object):

    def __init__(self,):

        self.proposal_sampling = TrajectorySampling(num_poses=60, interval_length=0.05)

        self._ego_vehicle_parameters=VehicleParameters(
                                vehicle_name="carla",
                                vehicle_type="carla",
                                width= 1.85,
                                front_length=2.042,
                                rear_length=2.042,
                                wheel_base=3.089,
                                cog_position_from_rear_axle=0,
                                height=1.777,
                            )
        self._motion_model = BatchKinematicBicycleModel()
        self._tracker = BatchLQRTracker()

    def control_lqr(self, waypoints, speed, acceleration):

        velocity=np.array([0,speed])
        acceleration=acceleration[:2]

        rel_waypoints=np.zeros_like(waypoints)

        rel_waypoints[:,-1]=waypoints[:,-1]-np.pi/2
        rel_waypoints[:,0]=waypoints[:,1]
        rel_waypoints[:,1]=waypoints[:,0]

        rear_axle_velocity_2d = StateVector2D(*velocity)
        rear_axle_acceleration_2d = StateVector2D(*acceleration)

        initial_ego_state=EgoState.build_from_rear_axle(
            StateSE2(0,0,np.pi/2),
            tire_steering_angle=np.pi/2,
            vehicle_parameters=self._ego_vehicle_parameters,
            time_point=TimePoint(0),
            rear_axle_velocity_2d=rear_axle_velocity_2d,
            rear_axle_acceleration_2d=rear_axle_acceleration_2d,
        )

        pred_trajectory = transform_trajectory(Trajectory(rel_waypoints,TrajectorySampling(time_horizon=3, interval_length=0.5)), initial_ego_state)

        pred_states = get_trajectory_as_array(pred_trajectory, self.proposal_sampling,
                                            initial_ego_state.time_point)

        states = pred_states[None]

        # simulated_states = self.simulator.simulate_proposals(trajectory_states, initial_ego_state)


        self._motion_model._vehicle = initial_ego_state.car_footprint.vehicle_parameters
        self._tracker._discretization_time = self.proposal_sampling.interval_length

        proposal_states = states[:, : self.proposal_sampling.num_poses + 1]
        self._tracker.update(proposal_states)

        # state array representation for simulated vehicle states
        simulated_states = np.zeros(proposal_states.shape, dtype=np.float64)
        simulated_states[:, 0] = ego_state_to_state_array(initial_ego_state)

        # timing objects
        current_time_point = initial_ego_state.time_point
        delta_time_point = TimeDuration.from_s(self.proposal_sampling.interval_length)

        current_iteration = SimulationIteration(current_time_point, 0)
        next_iteration = SimulationIteration(current_time_point + delta_time_point, 1)

        command_states = self._tracker.track_trajectory(
            current_iteration,
            next_iteration,
            simulated_states[:, 0],
        )

        acc=command_states[0][0]

        STEERING_RATE=command_states[0][1]

        # if acc<0:
        #     brake=brake
        # else:
        steer=0
        throttle=1
        brake=0
            

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
        }

        return steer, throttle, brake, metadata