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
#Slender man
# FIXED_UNITY_OFFSET = [
#     [0,"m_avg_Pelvis",0.00217014,1.458927,0.02859175],
#     [1,"m_avg_L_Hip",-0.05858135,-0.1234201,-0.01766408],
#     [2,"m_avg_R_Hip",0.06030973,-0.09051328,-0.01354253],
#     [3,"m_avg_Spine1",-0.004439451,0.1866054,-0.03838522],
#     [4,"m_avg_L_Knee",-0.04345143,-0.5797042,0.008037003],
#     [5,"m_avg_R_Knee",0.04325663,-0.5755318,-0.004843044],
#     [6,"m_avg_Spine2",-0.004488442,0.2069346,0.02682033],
#     [7,"m_avg_L_Ankle",0.01480183,-0.6402773,-0.03743253],
#     [8,"m_avg_R_Ankle",-0.01913394,-0.6300467,-0.03453969],
#     [9,"m_avg_Spine3",0.00226459,0.08404858,0.002855046],
#     [10,"m_avg_L_Foot",-0.04075197,-0.0393,0.1220452],
#     [11,"m_avg_R_Foot",0.03480373,-0.0654,0.1304466],
#     [12,"m_avg_Neck",0.01339018,0.274,-0.03346758],
#     [13,"m_avg_L_Collar",-0.1075537,0.1709996,-0.01889817],
#     [14,"m_avg_R_Collar",0.1244305,0.1687084,-0.02370739],
#     [15,"m_avg_Head",-0.01011724,0.1334229,0.0504067],
#     [16,"m_avg_L_Shoulder",-0.1843821,0.06780761,-0.019046],
#     [17,"m_avg_R_Shoulder",0.1698425,0.07027992,-0.008472068],
#     [18,"m_avg_L_Elbow",-0.3829978,-0.02347351,-0.02294649],
#     [19,"m_avg_R_Elbow",0.3901913,-0.02155391,-0.03126873],
#     [20,"m_avg_L_Wrist",-0.3986055,0.01908823,-0.007351562],
#     [21,"m_avg_R_Wrist",0.4037541,0.01021783,-0.005927166]
# ]

# Dwarf
FIXED_UNITY_OFFSET = [
    [0,"m_avg_Pelvis",0.001627605,0.5106243,0.02144381],
    [1,"m_avg_L_Hip",-0.04393601,-0.013,-0.01324806],
    [2,"m_avg_R_Hip",0.0452323,-0.02,-0.0101569],
    [3,"m_avg_Spine1",-0.003329588,0.0933027,-0.02878891],
    [4,"m_avg_L_Knee",-0.03258857,-0.249,0.041],
    [5,"m_avg_R_Knee",0.03244247,-0.2608,-0.003632283],
    [6,"m_avg_Spine2",-0.003366332,0.1034673,0.02011525],
    [7,"m_avg_L_Ankle",0.01110137,-0.3201386,-0.0280744],
    [8,"m_avg_R_Ankle",-0.01435046,-0.3150233,-0.02590477],
    [9,"m_avg_Spine3",0.001698443,0.04202429,0.002141285],
    [10,"m_avg_L_Foot",-0.03056398,-0.04528124,0.0915339],
    [11,"m_avg_R_Foot",0.0261028,-0.04648668,0.09783495],
    [12,"m_avg_Neck",0.01339018,0.188,-0.03346758],
    [13,"m_avg_L_Collar",-0.05377685,0.08549978,-0.01417363],
    [14,"m_avg_R_Collar",0.06221524,0.08435422,-0.01778054],
    [15,"m_avg_Head",-0.01011724,0.048,0.0504067],
    [16,"m_avg_L_Shoulder",-0.09219105,0.0339038,-0.0142845],
    [17,"m_avg_R_Shoulder",0.08492123,0.03513996,-0.006354051],
    [18,"m_avg_L_Elbow",-0.2042655,-0.01173676,-0.01720987],
    [19,"m_avg_R_Elbow",0.2341148,-0.01077695,-0.02345155],
    [20,"m_avg_L_Wrist",-0.265737,0.01272549,-0.007351562],
    [21,"m_avg_R_Wrist",0.2691694,0.006811886,-0.005927166]
]

UNITY_OFFSETS_NAME = np.array(FIXED_UNITY_OFFSET)[:, 1]
FIXED_UNITY_OFFSET = np.array(FIXED_UNITY_OFFSET)[:, 2:5].astype(np.float32)
FIXED_UNITY_OFFSET = torch.from_numpy(FIXED_UNITY_OFFSET).float()


class smpl_fk_fixed:
    def __init__(self, device = 'cpu'):
        self.n_joints = len(FIXED_UNITY_OFFSET)
        self.device = device
        self.n_joints = len(FIXED_UNITY_OFFSET)
        self.offsets = FIXED_UNITY_OFFSET.to(device)
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