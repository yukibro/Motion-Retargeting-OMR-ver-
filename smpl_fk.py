import torch
import numpy as np


if __name__ == '__main__':
    print('This file is not ment to be run as main, but to be imported')
    exit()

UNITY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

EXTRA_JOINT_PARENTS = [15, 20, 21]
EXTRA_JOINT_OFFSETS = torch.tensor([
    [0,0.09,0.06],
    [0,0,0],
    [0,0,0]
]).float()
EXTRA_JOINT_ROTMAT = torch.tensor([
    [[1,0,0],[0,1,0],[0,0,1]],
    [[1,0,0],[0,1,0],[0,0,1]],
    [[1,0,0],[0,1,0],[0,0,1]] 
]).float()
# identity matrix, for now
EXTRA_JOINTS_NAME = ['HMD', 'L_Controller', 'R_Controller']

FEET_JOINTS = [7, 8, 10, 11]
UNITY_OFFSETS = [
    [0,"m_avg_Pelvis",0.00217014,0.9726178,0.02859175],
    [1,"m_avg_L_Hip",-0.05858135,-0.08228004,-0.01766408],
    [2,"m_avg_R_Hip",0.06030973,-0.09051328,-0.01354253],
    [3,"m_avg_Spine1",-0.004439451,0.1244036,-0.03838522],
    [4,"m_avg_L_Knee",-0.04345143,-0.3864695,0.008037003],
    [5,"m_avg_R_Knee",0.04325663,-0.3836879,-0.004843044],
    [6,"m_avg_Spine2",-0.004488442,0.1379564,0.02682033],
    [7,"m_avg_L_Ankle",0.01480183,-0.4268515,-0.03743253],
    [8,"m_avg_R_Ankle",-0.01913394,-0.4200311,-0.03453969],
    [9,"m_avg_Spine3",0.00226459,0.05603239,0.002855046],
    [10,"m_avg_L_Foot",-0.04075197,-0.06037499,0.1220452],
    [11,"m_avg_R_Foot",0.03480373,-0.06198224,0.1304466],
    [12,"m_avg_Neck",0.01339018,0.2116355,-0.03346758],
    [13,"m_avg_L_Collar",-0.07170247,0.1139997,-0.01889817],
    [14,"m_avg_R_Collar",0.08295365,0.1124723,-0.02370739],
    [15,"m_avg_Head",-0.01011724,0.0889486,0.0504067],
    [16,"m_avg_L_Shoulder",-0.1229214,0.04520507,-0.019046],
    [17,"m_avg_R_Shoulder",0.1132283,0.04685328,-0.008472068],
    [18,"m_avg_L_Elbow",-0.2553319,-0.01564901,-0.02294649],
    [19,"m_avg_R_Elbow",0.2601275,-0.01436927,-0.03126873],
    [20,"m_avg_L_Wrist",-0.265737,0.01272549,-0.007351562],
    [21,"m_avg_R_Wrist",0.2691694,0.006811886,-0.005927166]
]

UNITY_OFFSETS_NAME = np.array(UNITY_OFFSETS)[:, 1]
UNITY_OFFSETS = np.array(UNITY_OFFSETS)[:, 2:5].astype(np.float32)
UNITY_OFFSETS = torch.from_numpy(UNITY_OFFSETS).float()

class smpl_fk:
    def __init__(self, device = 'cpu'):
        self.n_joints = len(UNITY_OFFSETS)
        self.device = device
        self.n_joints = len(UNITY_OFFSETS)
        self.offsets = UNITY_OFFSETS.to(device)
        self.ex_offsets = EXTRA_JOINT_OFFSETS.to(device)
        self.ex_rot = EXTRA_JOINT_ROTMAT.to(device)
        
        self.device = device
    def get_feet_joints(self):
        return FEET_JOINTS
    def get_joint_names(self):
        return UNITY_OFFSETS_NAME
    def get_extra_joint_names(self):
        return EXTRA_JOINTS_NAME
    
    def fk_batch(self, root_trans, poses, joint_num = 22, is_neckshoulder_2 = False):

        self.offsets = self.offsets.detach()
        self.ex_offsets = self.ex_offsets.detach()
        self.ex_rot = self.ex_rot.detach()

        global_positions, global_rotations = torch.zeros((poses.shape[0], joint_num, 3), device = self.device), torch.zeros((poses.shape[0], joint_num, 3, 3), device = self.device)
        extra_positions, extra_rotations = torch.zeros((poses.shape[0], 3, 3), device = self.device), torch.zeros((poses.shape[0], 3, 3, 3), device = self.device)

        poses_copy_ = poses.clone()
        poses_copy = poses_copy_

        for joint in range(joint_num):
            parent_j = UNITY_PARENTS[joint]

            if parent_j == -1:
                if root_trans != None:
                    global_positions[:, joint] = root_trans
                    global_rotations[:, joint] = poses_copy[:, joint]
                else:
                    init_trans = torch.zeros((poses.shape[0], 3), device = self.device)
                    global_rotations[:, joint] = poses_copy[:, joint]
            else:
                global_positions[:, joint] = torch.matmul(global_rotations[:, parent_j], self.offsets[joint].unsqueeze(-1)).squeeze(-1) + global_positions[:, parent_j]
                global_rotations[:, joint] = torch.matmul(global_rotations[:, parent_j].clone(), poses_copy[:, joint]) #여기서 문제 발생
            
        for joint in range(self.ex_offsets.shape[0]):
            parent_j = EXTRA_JOINT_PARENTS[joint]
            extra_positions[:, joint] = torch.matmul(global_rotations[:, parent_j], self.ex_offsets[joint].unsqueeze(-1)).squeeze(-1) + global_positions[:, parent_j]
            extra_rotations[:, joint] = torch.matmul(global_rotations[:, parent_j], self.ex_rot[joint])

        return (global_positions, global_rotations), (extra_positions, extra_rotations)

    def fk_w_extra(self, root_trans = None, pose = None):
        global_position, global_rotation = self.fk_frame(root_trans, pose)
        extra_position = torch.zeros((len(EXTRA_JOINTS_NAME), 3)).to(self.device)
        extra_rotation = torch.zeros((len(EXTRA_JOINTS_NAME), 3, 3)).to(self.device)
        for i in range(len(EXTRA_JOINT_ROTMAT)):
            extra_position[i] = torch.matmul(global_rotation[EXTRA_JOINT_PARENTS[i]], EXTRA_JOINT_OFFSETS[i].to(self.device))
            extra_position[i] = extra_position[i] + global_position[EXTRA_JOINT_PARENTS[i]]
            extra_rotation[i] = torch.matmul(global_rotation[EXTRA_JOINT_PARENTS[i]], EXTRA_JOINT_ROTMAT[i].to(self.device))
        
        return (global_position, global_rotation), (extra_position, extra_rotation)