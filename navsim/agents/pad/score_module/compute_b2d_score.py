import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.creation import linestrings
from shapely import Point, creation


def compute_corners(boxes):
    # Calculate half dimensions
    x = boxes[:, 0]        # x-coordinate of the center
    y = boxes[:, 1]        # y-coordinate of the center
    half_width = boxes[:, 2]  / 2
    half_length = boxes[:, 3]  / 2
    headings= boxes[:, 4]

    cos_yaw = np.cos(headings)[...,None]
    sin_yaw = np.sin(headings)[...,None]

    # Compute the four corners
    corners_x = np.stack([half_length, half_length, -half_length, -half_length],axis=-1)
    corners_y = np.stack([half_width, -half_width, -half_width, half_width],axis=-1)

    # Rotate corners by yaw
    rot_corners_x = cos_yaw * corners_x + (-sin_yaw) * corners_y
    rot_corners_y = sin_yaw * corners_x + cos_yaw * corners_y

    # Translate corners to the center of the bounding box
    corners = np.stack((rot_corners_x + x[...,None], rot_corners_y + y[...,None]), axis=-1)

    return corners

ego_width, ego_length = 0.925*2, 2.042*2

def eval_single(ego_corners,fut_corners,fut_mask):

    agent_number=fut_corners.shape[0]
    n_future=fut_corners.shape[1]

    ttc_collsion=False
    ttc_agent=np.zeros([6,4,2])
    ttc_agent_mask=np.zeros([6])

    for t in range(n_future):
        ego_poly=Polygon([(point[0], point[1]) for point in ego_corners[t][0]])

        if not ttc_collsion:
            ego_poly1=Polygon([(point[0], point[1]) for point in ego_corners[t][1]])

            #ego_poly2=Polygon([(point[0], point[1]) for point in ego_corners[t][2]])

        for n in range(agent_number):
            if fut_mask[n][t]:
                fut_corners_tn=fut_corners[n][t]
                box_poly = Polygon([(point[0], point[1]) for point in fut_corners_tn])
                collision = ego_poly.intersects(box_poly)
                if collision:
                    collision_agent=fut_corners[n][:t+1]
                    collision_agent_mask=fut_mask[n][:t+1]
                    return True,ttc_collsion,collision_agent,collision_agent_mask,ttc_agent,ttc_agent_mask
                elif not ttc_collsion:
                    ttc_collsion = ego_poly1.intersects(box_poly) #(ego_poly1.intersects(box_poly)) # |(ego_poly2.intersects(box_poly))
                    if ttc_collsion:
                        ttc_agent=fut_corners[n][:t+1]
                        ttc_agent_mask=fut_mask[n][:t+1]

    return False,ttc_collsion,np.zeros([6,4,2]), np.zeros([6]),ttc_agent,ttc_agent_mask

def evaluate_coll( proposals,fut_corners,fut_mask):
    n_future = proposals.shape[1]

    proposals1=np.concatenate([np.zeros_like(proposals[:,:1,:2]),proposals[:,:,:2]],axis=1)

    vel=proposals1[:,1:,:2]-proposals1[:,:-1,:2]

    proposals_05=np.concatenate([proposals[:,:,:2]+vel,proposals[:,:,2:]],axis=-1)

    #proposals_10=np.concatenate([proposals[:,:,:2]+vel*2, proposals[:,:,2:]],axis=-1)

    proposals=np.stack([proposals,proposals_05],axis=2)#,proposals_10

    n_proposal = proposals.shape[0]

    ego_box = np.zeros((n_proposal,n_future,proposals.shape[2], 5))

    heading= proposals[...,2]

    ego_box[..., 0] = proposals[...,0]+0.39*np.cos(heading)
    ego_box[..., 1] = proposals[...,1]+0.39*np.sin(heading)
    ego_box[..., 2] = ego_width
    ego_box[..., 3] = ego_length
    ego_box[..., 4] = proposals[...,2]

    ego_corners=compute_corners(ego_box.reshape(-1,5)).reshape(n_proposal,n_future,proposals.shape[2],4,2)

    num_col=2

    key_agent_corners = np.zeros([n_proposal,num_col, 6, 4, 2])
    key_agent_labels = np.zeros([n_proposal,num_col, 6],dtype=bool)
    collision_all = np.zeros([n_proposal,n_future])
    ttc_collision_all = np.zeros([n_proposal,n_future])

    for i in range(n_proposal):
        collision_all_i,ttc_collision_i,collision_agent_i,collision_agent_mask_i,ttc_agent_i,ttc_agent_mask_i=eval_single(ego_corners[i],fut_corners,fut_mask)
        collision_all[i]=collision_all_i
        ttc_collision_all[i]=ttc_collision_i
        key_agent_corners[i,0,:len(collision_agent_i)]=collision_agent_i
        key_agent_labels[i,0,:len(collision_agent_mask_i)]=collision_agent_mask_i
        key_agent_corners[i,1,:len(ttc_agent_i)]=ttc_agent_i
        key_agent_labels[i,1,:len(ttc_agent_mask_i)]=ttc_agent_mask_i

    return collision_all,ttc_collision_all,key_agent_corners,key_agent_labels

def get_scores(args):

    return [get_sub_score(a["token"],a["poses"],a["target_trajectory"],a["lidar2world"],a["nearby_point"]) for a in args]

def get_sub_score( fut_corners,proposals,target_trajectory,lidar2world,nearby_point):

    proposals_xy=proposals[..., :2]

    z = lidar2world[2, 3]

    proposals_xyz=np.concatenate([proposals_xy,np.zeros_like(proposals_xy[...,:1])+z,np.ones_like(proposals_xy[...,:1])],axis=-1)

    global_proposals =np.einsum("ij,ntj->nti",lidar2world,proposals_xyz)[...,:2]

    #global_proposals = proposals[..., :2] + xy[None, None]

    #nearby_point=np.concatenate([target_trajectory[...,:2]+xy,nearby_point],axis=0)

    dist = np.linalg.norm(global_proposals[None] - nearby_point[:, None, None], axis=-1)

    min_dist = dist.min(0)

    # l2=np.linalg.norm(proposals[...,:2] - target_trajectory[...,:2] [ None],axis=-1)

    # l2_2s=l2[:,:4].mean(-1)

    # progress=np.exp(-l2_2s/5)

    l1= np.linalg.norm(proposals - target_trajectory[ None],ord=1,axis=-1).mean(-1)
    
    min_index=np.argmin(l1,axis=0)
    
    progress =np.zeros([len(proposals)])
    
    progress[min_index]=1

    # target_line=np.concatenate([np.zeros([1,2]),target_trajectory[...,:2]])
    #
    # centerline=linestrings(target_line)
    #
    # progress_in_meter=np.zeros([len(proposals)])
    #
    # for proposal_idx,proposal in enumerate(proposals[...,:2]):
    #     start_point = Point(proposal[0])
    #     end_point = Point(proposal[-1])
    #     progress = centerline.project([start_point, end_point])
    #     progress_in_meter[proposal_idx] = progress[1] - progress[0]
    #
    # progress = np.clip(progress_in_meter, a_min=0, a_max=1)

    on_road = min_dist < 3.5

    ego_areas=min_dist[:,:,None]>np.arange(2,3.5,0.1)[None,None] #np.stack([min_dist>4,min_dist>3,min_dist>2],axis=-1)

    drivable_area_compliance=on_road.all(-1)

    fut_mask=fut_corners.all(-1).all(-1)

    collsions,ttc_collision,key_agent_corners,key_agent_labels=evaluate_coll(proposals,fut_corners,fut_mask)

    collision=1-collsions.any(-1)

    ttc=1-ttc_collision.any(-1)

    proposals=np.concatenate([np.zeros_like(proposals[:,:1,:2]),proposals[:,:,:2]],axis=1)

    vel=(proposals[:,1:,:2]-proposals[:,:-1,:2])/0.5

    acc=np.linalg.norm(vel[:,1:]-vel[:,:-1],axis=-1)/0.5

    angle=np.arctan2(vel[:,:,1], vel[:,:,0])

    heading=np.concatenate([np.zeros_like(angle[:,:1])+np.pi/2,angle],axis=1)

    yaw_rate=(heading[:,1:]-heading[:,:-1])/0.5
    
    yaw_accel=(yaw_rate[:,1:]-yaw_rate[:,:-1])/0.5

    desired_speed=np.linalg.norm(vel,axis=-1).mean(-1)
    
    comfort=(acc<8).all(-1) & (desired_speed<15) & (np.abs(yaw_rate)<1).all(-1) & (np.abs(yaw_accel)<2).all(-1)

    progress=collision*drivable_area_compliance*progress

    final_scores=collision*drivable_area_compliance*(ttc*5/12+progress*5/12+comfort*2/12)

    target_scores=np.stack([collision,drivable_area_compliance,progress,ttc,comfort,final_scores],axis=-1)

    #print(target_scores[0])
    # if target_scores.mean()!=1:
    #     print(1)

    return target_scores,key_agent_corners,key_agent_labels,ego_areas


