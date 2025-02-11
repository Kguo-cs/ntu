import carla
import numpy as np

import cv2

from pathlib import Path
import os
import argparse
import time
import subprocess
import json
import sys
import pickle

CARLA_ROOT=os.environ.get("CARLA_ROOT")

sys.path.append(CARLA_ROOT + "/PythonAPI")
sys.path.append(CARLA_ROOT + "/PythonAPI/carla")

sys.path.append(CARLA_ROOT + "/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")



def check_waypoints_status(waypoints_list):
    first_wp = waypoints_list[0]
    init_status = first_wp.is_junction
    current_status = first_wp.is_junction
    change_status_time = 0
    for wp in waypoints_list[1:]:
        if wp.is_junction != current_status:
            current_status = wp.is_junction
            change_status_time += 1
        pass
    if change_status_time == 0:
        return 'Junction' if init_status else 'Normal'
    elif change_status_time == 1:
        return 'EnterNormal' if init_status else 'EnterJunction'
    elif change_status_time == 2:
        return 'PassNormal' if init_status else 'PassJunction'
    else:
        return 'StartJunctionMultiChange' if init_status else 'StartNormalMultiChange'


class TriggerVolumeGettor(object):

    @staticmethod
    def get_global_bbx(actor, bbx):
        if actor.is_alive:
            bbx.location = actor.get_transform().transform(bbx.location)
            bbx.rotation = actor.get_transform().rotation
            return bbx
        return None

    @staticmethod
    def get_corners_from_actor_list(actor_list):
        for actor_transform, bb_loc, bb_ext in actor_list:
            corners = [carla.Location(x=-bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=-bb_ext.y),
                       carla.Location(x=bb_ext.x, y=0),
                       carla.Location(x=bb_ext.x, y=bb_ext.y),
                       carla.Location(x=-bb_ext.x, y=bb_ext.y)]
            corners = [bb_loc + corner for corner in corners]

            corners = [actor_transform.transform(corner) for corner in corners]
            corners = [[corner.x, corner.y, corner.z] for corner in corners]
        return corners

    @staticmethod
    def insert_point_into_dict(lane_marking_dict, corners, road_id, parent_actor_location, Volume_Type=None):
        if road_id not in lane_marking_dict.keys():
            print("Cannot find road:", road_id)
            raise
        if Volume_Type is None:
            print("Missing 'Volume Type' ")
            raise
        if 'Trigger_Volumes' not in lane_marking_dict[road_id]:
            lane_marking_dict[road_id]['Trigger_Volumes'] = [
                {'Points': corners[:], 'Type': Volume_Type, 'ParentActor_Location': parent_actor_location[:]}]
        else:
            lane_marking_dict[road_id]['Trigger_Volumes'].append(
                {'Points': corners[:], 'Type': Volume_Type, 'ParentActor_Location': parent_actor_location[:]})

    @staticmethod
    def get_stop_sign_trigger_volume(all_stop_sign_actors, lane_marking_dict, carla_map):
        for actor in all_stop_sign_actors:
            bb_loc = carla.Location(actor.trigger_volume.location)
            bb_ext = carla.Vector3D(actor.trigger_volume.extent)
            bb_ext.x = max(bb_ext.x, bb_ext.y)
            bb_ext.y = max(bb_ext.x, bb_ext.y)
            base_transform = actor.get_transform()
            stop_info_list = [(carla.Transform(base_transform.location, base_transform.rotation), bb_loc, bb_ext)]
            corners = TriggerVolumeGettor.get_corners_from_actor_list(stop_info_list)

            trigger_volume_wp = carla_map.get_waypoint(base_transform.transform(bb_loc))
            actor_loc = actor.get_location()
            actor_loc_points = [actor_loc.x, actor_loc.y, actor_loc.z]
            TriggerVolumeGettor.insert_point_into_dict(lane_marking_dict, corners, trigger_volume_wp.road_id,
                                                       actor_loc_points, Volume_Type='StopSign')

        pass

    @staticmethod
    def get_traffic_light_trigger_volume(all_trafficlight_actors, lane_marking_dict, carla_map):
        for actor in all_trafficlight_actors:
            base_transform = actor.get_transform()
            tv_loc = actor.trigger_volume.location
            tv_ext = actor.trigger_volume.extent
            x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)
            area = []
            for x in x_values:
                point_location = base_transform.transform(tv_loc + carla.Location(x=x))
                area.append(point_location)
            ini_wps = []
            for pt in area:
                wpx = carla_map.get_waypoint(pt)
                # As x_values are arranged in order, only the last one has to be checked
                if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
                    ini_wps.append(wpx)

            close2junction_points = []
            littlefar2junction_points = []
            for wpx in ini_wps:
                while not wpx.is_intersection:
                    next_wp = wpx.next(0.5)
                    if not next_wp:
                        break
                    next_wp = next_wp[0]
                    if next_wp and not next_wp.is_intersection:
                        wpx = next_wp
                    else:
                        break
                vec_forward = wpx.transform.get_forward_vector()
                vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)  # 2D

                loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
                loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
                close2junction_points.append([loc_left.x, loc_left.y, loc_left.z])
                close2junction_points.append([loc_right.x, loc_right.y, loc_right.z])

                try:
                    loc_far_left = wpx.previous(0.5)[0].transform.location - 0.4 * wpx.lane_width * vec_right
                    loc_far_right = wpx.previous(0.5)[0].transform.location + 0.4 * wpx.lane_width * vec_right
                except Exception:
                    continue

                littlefar2junction_points.append([loc_far_left.x, loc_far_left.y, loc_far_left.z])
                littlefar2junction_points.append([loc_far_right.x, loc_far_right.y, loc_far_right.z])

            traffic_light_points = close2junction_points + littlefar2junction_points[::-1]
            trigger_volume_wp = carla_map.get_waypoint(base_transform.transform(tv_loc))
            actor_loc = actor.get_location()
            actor_loc_points = [actor_loc.x, actor_loc.y, actor_loc.z]
            TriggerVolumeGettor.insert_point_into_dict(lane_marking_dict, traffic_light_points,
                                                       trigger_volume_wp.road_id, actor_loc_points,
                                                       Volume_Type='TrafficLight')
        pass

    pass


t = 0


class LankMarkingGettor(object):
    '''
        structure of lane_marking_dict:
        {
            road_id_0: {
                lane_id_0: [{'Points': [((location.x,y,z) array, (rotation.roll, pitch, yaw))], 'Type': 'lane_marking_type', 'Color':'color', 'Topology':[neighbor array]}, ...]
                ... ...
                'Trigger_Volumes': [{'Points': [(location.x,y,z) array], 'Type': 'trigger volume type', 'ParentActor_Location': (location.x,y,z)}]
            }
            ... ...
        }
        "location array" is an array formed as (location_x, location_y, location_z) ...
        'lane_marking_type' is string of landmarking type, can be 'Broken', 'Solid', 'SolidSolid', 'Other', 'NONE', etc.
        'color' is string of landmarking color, can be 'Blue', 'White', 'Yellow',  etc.
         neighbor array contains the ('road_id', 'lane_id') of the current landmarking adjacent to, it is directional.
         and if current 'Type' == 'Center', there will exist a 'TopologyType' key which record the current lane's topology status.
         if there exist a trigger volume in current road, key 'Trigger_Volumes' will be added into dict
         where 'Points' refer to the vertexs location array, 'Type' can be 'StopSign' or 'TrafficLight'
         'ParentActor_Location' is the location of parent actor relevant to this trigger volume.
    '''

    @staticmethod
    def get_lanemarkings(carla_map, lane_marking_dict={}, pixels_per_meter=2, precision=0.05):

        topology = [x[0] for x in carla_map.get_topology()]
        topology = sorted(topology, key=lambda w: w.road_id)

        map_list=[]

        for waypoint in topology:
            waypoints = [waypoint]
            # Generate waypoints of a road id. Stop when road id differs
            nxt = waypoint.next(precision)
            if len(nxt) > 0:
                nxt = nxt[0]
                temp_wp = nxt
                while nxt.road_id == waypoint.road_id:
                    waypoints.append(nxt)
                    nxt = nxt.next(precision)
                    if len(nxt) > 0:
                        nxt = nxt[0]
                    else:
                        break
            print("current road id: ", waypoint.road_id)
            print("lane id:", waypoint.lane_id)
            maps=[]
            for waypoint in waypoints:
                w_transform=waypoint.transform
                road_lane_id=waypoint.road_id+waypoint.lane_id*0.001
                maps.append((w_transform.location.x,w_transform.location.y,waypoint.lane_width*0.5,road_lane_id))

            maps=np.array(maps).astype(np.float32)[::10]

            map_list.append(maps)

        all_map=np.concatenate(map_list)

        return all_map



if __name__ == '__main__':
    map_dict={}
    cmd1 = f"{os.path.join(CARLA_ROOT, 'CarlaUE4.sh')} -RenderOffScreen -nosound -carla-rpc-port=20001"
    server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
    print(cmd1, server.returncode, flush=True)
    time.sleep(10)
    client = carla.Client('localhost', 20001)
    client.set_timeout(300)

    for id in ['01','02','03','04','05','06','07','10HD','11','12','13','15']:
        carla_town = 'Town'+id

        world = client.load_world(carla_town)
        print("******** sucessfully load the town:", carla_town, " ********")
        carla_map = world.get_map()

        arr = LankMarkingGettor.get_lanemarkings(world.get_map())
        print("****** get all lanemarkings ******")

        map_dict[carla_town[:6]]=arr

    with open(os.getenv('NAVSIM_EXP_ROOT') + "/map.pkl", 'wb') as f:
        pickle.dump(map_dict, f)
