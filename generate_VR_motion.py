import numpy as np
import torch
import time
import os 
from models.select_model import define_Model
from utils.utils_transform import matrot2sixd, sixd2matrot, sixd2aa, aa2matrot, matrot2aa
from utils import smpl_fk_fixed, utils_option as option
from utils import smpl_fk
import argparse
from OMR_edit import OMR

start = time.time()
device = "cuda"
fk_engine = smpl_fk.smpl_fk(device=device)
f_fk_engine = smpl_fk_fixed.smpl_fk_fixed(device=device)

RADIAN = 180 / np.pi
DEGREE = np.pi / 180
SMPL_HEIGHT = 1.591594
SMPL_ARM_LENGTH = 1.4411717

SMPL_HIERARCHY = {
    'pelvis': None,
    'l_Hip': 'pelvis',
    'r_Hip': 'pelvis',
    'spine1': 'pelvis',
    'l_Knee': 'l_Hip',
    'r_Knee': 'r_Hip',
    'spine2': 'spine1',
    'l_Ankle': 'l_Knee',
    'r_Ankle': 'r_Knee',
    'spine3': 'spine2',
    'l_Foot': 'l_Ankle',
    'r_Foot': 'r_Ankle',
    'neck': 'spine3',
    'l_Collar': 'spine3',
    'r_Collar': 'spine3',
    'head': 'neck',
    'l_Shoulder': 'l_Collar',
    'r_Shoulder': 'r_Collar',
    'l_Elbow': 'l_Shoulder',
    'r_Elbow': 'r_Shoulder',
    'l_Wrist': 'l_Elbow',
    'r_Wrist': 'r_Elbow'
}

joint_names = [
    'pelvis', 'l_Hip', 'r_Hip', 'spine1', 'l_Knee', 'r_Knee', 'spine2',
    'l_Ankle', 'r_Ankle', 'spine3', 'l_Foot', 'r_Foot', 'neck', 'l_Collar',
    'r_Collar', 'head', 'l_Shoulder', 'r_Shoulder', 'l_Elbow', 'r_Elbow',
    'l_Wrist', 'r_Wrist'
]

save_unity_npz = True
support_dir = "./support_data/"
subject_gender = "male"
bm_fname = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format(subject_gender))
dmpl_fname = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format(subject_gender))
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters
# bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
json_path='options/opt_ajlm.json'

parser = argparse.ArgumentParser()
parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
parser.add_argument('--task', type=str, default='AvatarJLM', help='Experiment name.')
parser.add_argument('--checkpoint', type=str, required=True, help='Trained model weights.')
args = parser.parse_args()
opt = option.parse(args.opt, args, is_train=False)

paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
if isinstance(paths, str):
    if not os.path.exists(paths):
        os.makedirs(paths)
else:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
init_iter, init_path_G = option.find_last_checkpoint(opt['path']['pretrained_netG'], net_type='G')
current_step = init_iter

option.save(opt)

opt = option.dict_to_nonedict(opt)

model = define_Model(opt)
model.init_test()

with open("./VR_Capture/hands_motion_wo.txt", 'r') as file:
        content = file.read()
matrix_strings = content.split(',')
record_data = []
for matrix_string in matrix_strings:
        matrix_values = list(map(float, matrix_string.split()))
        
        if len(matrix_values) != 16:
            print(f"Invalid matrix length: {len(matrix_values)}, expected 16")
            continue
        matrix = np.array(matrix_values).reshape((4, 4))
        record_data.append(matrix)

record_data_tensor = torch.tensor(record_data).to(device)
record_data_tensor = record_data_tensor.reshape(-1,3,4,4)
vr_position = record_data_tensor[:,:,:3,3]

if torch.absolute(vr_position[0,0,1]) == 0: 
    height_scale_ratio = 1
else:
    height_scale_ratio = SMPL_HEIGHT / torch.absolute(vr_position[0,0,1])

if torch.absolute(vr_position[0,1,0] - vr_position[0,2,0]) == 0:
    arm_scale_ratio = 1
else: 
    arm_scale_ratio = SMPL_ARM_LENGTH / 1.718


vr_rotation = record_data_tensor[:,:,:3,:3]

generate_output_name = "hands_dwarf_wo"
generate_output_dir = "./VR_offline_output/" + generate_output_name + "/"

batch_size = 1
window_size = 41

###### Get input x from VR_npz_file ######
input_pos = vr_position.reshape(-1,9).float() # frame, 9
input_rotmat = vr_rotation.reshape(-1,27).float() # frame, 27

amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.0]]]).to(device)
amass_rot_inv = torch.inverse(amass_rot).float()

input_original_trans = input_pos.reshape(-1, 3, 3).clone()
input_unity_trans = torch.zeros_like(input_original_trans)
input_unity_trans[:, :, 0] = -input_original_trans[:, :, 0]
input_unity_trans[:, :, 1] = input_original_trans[:, :, 1]
input_unity_trans[:, :, 2] = input_original_trans[:, :, 2]

input_original_aa = matrot2aa(input_rotmat.reshape(-1, 3, 3)).reshape(-1, 3, 3)
input_unity_aa = torch.zeros_like(input_original_aa)
input_unity_aa[:,:, 0] = input_original_aa[:, :, 0]
input_unity_aa[:,:, 1] = -input_original_aa[:, :, 1]
input_unity_aa[:,:, 2] = -input_original_aa[:, :, 2]

input_unity_rotmat = aa2matrot(input_unity_aa.reshape(-1, 3)).reshape(-1, 3, 3, 3)

input_unity_rotmat = amass_rot_inv.matmul(input_unity_rotmat.reshape(-1, 3, 3)).reshape(-1, 3, 3, 3)
input_unity_trans[:,0] = amass_rot_inv.matmul(input_unity_trans[:,0].unsqueeze(-1)).reshape(-1,3)
input_unity_trans[:,1] = amass_rot_inv.matmul(input_unity_trans[:,1].unsqueeze(-1)).reshape(-1,3)
input_unity_trans[:,2] = amass_rot_inv.matmul(input_unity_trans[:,2].unsqueeze(-1)).reshape(-1,3)


num_frames = input_pos.shape[0]
input_rotmat_6D = matrot2sixd(input_unity_rotmat.reshape(-1,3,3)).reshape(-1, 3, 6)
input_rotvel = torch.matmul(torch.inverse(input_unity_rotmat.reshape(-1, 3, 3, 3)[:-1]), input_unity_rotmat.reshape(-1, 3, 3, 3)[1:])
input_rotvel_6D = matrot2sixd(input_rotvel.reshape(-1,3,3)).reshape(-1, 3, 6)
input_posvel = (input_unity_trans.reshape(num_frames, 3, 3)[1:] - input_unity_trans.reshape(num_frames, 3, 3)[:-1])

input = torch.cat([input_rotmat_6D[1:].reshape(num_frames-1, -1), input_rotvel_6D.reshape(num_frames-1, -1), input_unity_trans[1:].reshape(num_frames-1, -1), input_posvel.reshape(num_frames-1, -1)], dim=-1,)
gara_input = torch.ones(num_frames-1,22,18)

gara_input[:,15,:6] = input[:,:6] # rotation
gara_input[:,15,6:12] = input[:,18:24] # angular velocity
gara_input[:,15,12:15] = input[:,36:39] # position
gara_input[:,15,15:18] = input[:,45:48] # velocity

gara_input[:,20,:6] = input[:, 6:12] # rotation
gara_input[:,20,6:12] = input[:, 24:30] # angular velocity
gara_input[:,20,12:15] = input[:,39:42] # position
gara_input[:,20,15:18] = input[:,48:51] # velocity

gara_input[:,21,:6] = input[:, 12:18] # rotation
gara_input[:,21,6:12] = input[:, 30:36] # angular velocity
gara_input[:,21,12:15] = input[:,42:45] # position
gara_input[:,21,15:18] = input[:,51:54] # velocity

num_frames = num_frames - 1
##########################################
model.netG.eval()
with torch.no_grad():
        if gara_input.shape[0] <= window_size:
                pred_global_orientation_list = []  
                pred_joint_rotation_list = []
                for frame_idx in range(0, gara_input.shape[0]):
                    outputs = model.netG(gara_input[0:frame_idx+1].unsqueeze(0))
                    pred_global_orientation, pred_joint_rotation = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list.append(pred_global_orientation)
                    pred_joint_rotation_list.append(pred_joint_rotation)
                pred_global_orientation_tensor = torch.cat(pred_global_orientation_list, dim=0)
                pred_joint_rotation_tensor = torch.cat(pred_joint_rotation_list, dim=0)
        else:
                input_list_2 = []
                pred_global_orientation_list_1 = []  
                pred_joint_rotation_list_1 = []        
                for frame_idx in range(0, window_size):
                    outputs = model.netG(gara_input[0:frame_idx+1].unsqueeze(0))
                    pred_init_pose = outputs['pred_init_pose'] # 66
                    pred_global_orientation = outputs['pred_global_orientation'] # 6
                    pred_joint_rotation = outputs['pred_joint_rotation'] # 126
                    pred_joint_position = outputs['pred_joint_position']
                    pred_global_position = outputs['pred_global_position']

                    pred_global_orientation, pred_joint_rotation = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list_1.append(pred_global_orientation)
                    pred_joint_rotation_list_1.append(pred_joint_rotation)
                pred_global_orientation_1 = torch.cat(pred_global_orientation_list_1, dim=0)
                pred_joint_rotation_1 = torch.cat(pred_joint_rotation_list_1, dim=0)
                    
                for frame_idx in range(window_size, gara_input.shape[0]):
                    input_list_2.append(gara_input[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0))
                input_tensor_2 = torch.cat(input_list_2, dim = 0)

                part_size = 30
                part_num = (input_tensor_2.shape[0] - 1) // part_size + 1
                pred_global_orientation_list = [pred_global_orientation_1]
                pred_joint_rotation_list = [pred_joint_rotation_1]
                for part_idx in range(part_num):
                    outputs = model.netG(input_tensor_2[part_size * part_idx:min(part_size * (part_idx+1), input_tensor_2.shape[0])])
                    pred_global_orientation_this_part, pred_joint_rotation_this_part = outputs['pred_global_orientation'][-1][:, -1], outputs['pred_joint_rotation'][-1][:, -1]
                    pred_global_orientation_list.append(pred_global_orientation_this_part)
                    pred_joint_rotation_list.append(pred_joint_rotation_this_part)
                pred_global_orientation_tensor = torch.cat(pred_global_orientation_list, dim=0)
                pred_joint_rotation_tensor = torch.cat(pred_joint_rotation_list, dim=0)

output_all = torch.cat([pred_global_orientation_tensor, pred_joint_rotation_tensor], dim=-1).float()
pred_fullbody_6d = output_all.detach().cuda().squeeze() # frame, 132

pred_fullbody_rotmat = sixd2matrot(pred_fullbody_6d.reshape(-1, 6)).reshape(-1, 22, 3, 3) # frame, 22, 3, 3
pred_fullbody_aa = sixd2aa(pred_fullbody_6d.reshape(-1, 6)).reshape(-1, 66) # frame, 66
t_head2world = input_unity_trans[1:, 0].clone()

(local_position, _),(_,_) = fk_engine.fk_batch(None,pred_fullbody_rotmat)

t_head2root = -local_position[:, 15, :]
t_root2world = t_head2root + t_head2world

(predicted_position, _),(_,_) = fk_engine.fk_batch(t_root2world,pred_fullbody_rotmat)

pred_original_trans = t_root2world
pred_original_trans = amass_rot.matmul(pred_original_trans.unsqueeze(-1)).view_as(pred_original_trans)
pred_unity_trans = torch.zeros_like(pred_original_trans)

pred_original_aa = pred_fullbody_aa.reshape(-1, 22, 3)
pred_fullbody_rotmat = aa2matrot(pred_fullbody_aa.reshape(-1, 3)).reshape(-1, 22, 3, 3)
pred_unity_rotmat = pred_fullbody_rotmat.clone()
pred_unity_rotmat[:,0] = amass_rot.matmul(pred_fullbody_rotmat[:,0].reshape(-1, 3, 3))
pred_original_aa = matrot2aa(pred_unity_rotmat.reshape(-1, 3, 3)).reshape(-1, 22, 3)

pred_unity_aa = torch.zeros_like(pred_original_aa)

pred_unity_trans[:, 0] = -pred_original_trans[:, 0]
pred_unity_trans[:, 1] = pred_original_trans[:, 1]
pred_unity_trans[:, 2] = pred_original_trans[:, 2]

pred_unity_aa[:,:, 0] = pred_original_aa[:, :, 0]
pred_unity_aa[:,:, 1] = -pred_original_aa[:, :, 1]
pred_unity_aa[:,:, 2] = -pred_original_aa[:, :, 2]

pred_unity_aa = pred_unity_aa.reshape(-1, 66)

pred_unity_rotmat = aa2matrot(pred_unity_aa.reshape(-1, 3)).reshape(-1, 22, 3, 3)

(pred_unity_pos, _),(_,_) = fk_engine.fk_batch(pred_unity_trans,pred_unity_rotmat)
(fixed_unity_pos, _),(_,_) = f_fk_engine.fk_batch(pred_unity_trans,pred_unity_rotmat)

pred_unity_trans = pred_unity_trans.detach().cpu()
pred_unity_rotmat = pred_unity_rotmat.detach().cpu()
pred_unity_pos = pred_unity_pos.detach().cpu()
pred_unity_aa = pred_unity_aa.reshape(-1,22,3).detach().cpu()
vr_position = vr_position.detach().cpu()
fixed_unity_pos = fixed_unity_pos.detach().cpu()
pred_unity_pos *= 100.0
fixed_unity_pos *= 100.0


targetList = ["l_Foot","r_Foot"]
endEffectorList = ["l_Foot","r_Foot"]

joint_data = {
    "targets": {
        "l_Foot" : pred_unity_pos[0, 10],
        "r_Foot" : pred_unity_pos[0, 11]
         },
    "positions": {
        "pelvis": fixed_unity_pos[0, 0],
        "l_Hip": fixed_unity_pos[0, 1],
        "r_Hip": fixed_unity_pos[0, 2],
        "spine1": fixed_unity_pos[0, 3],
        "l_Knee": fixed_unity_pos[0, 4],
        "r_Knee": fixed_unity_pos[0, 5],
        "spine2": fixed_unity_pos[0, 6],
        "l_Ankle": fixed_unity_pos[0, 7],
        "r_Ankle": fixed_unity_pos[0, 8],
        "spine3": fixed_unity_pos[0, 9],
        "l_Foot": fixed_unity_pos[0, 10],
        "r_Foot": fixed_unity_pos[0, 11],
        "neck": fixed_unity_pos[0, 12],
        "l_Collar": fixed_unity_pos[0, 13],
        "r_Collar": fixed_unity_pos[0, 14],
        "head": fixed_unity_pos[0, 15],
        "l_Shoulder": fixed_unity_pos[0, 16],
        "r_Shoulder": fixed_unity_pos[00, 17],
        "l_Elbow": fixed_unity_pos[0, 18],
        "r_Elbow": fixed_unity_pos[0, 19],
        "l_Wrist": fixed_unity_pos[0, 20],
        "r_Wrist": fixed_unity_pos[0, 21]
        },
    "angles": {
        "pelvis": pred_unity_aa[0, 0],
        "l_Hip": pred_unity_aa[0, 1],
        "r_Hip": pred_unity_aa[0, 2],
        "spine1": pred_unity_aa[0, 3],
        "l_Knee": pred_unity_aa[0, 4],
        "r_Knee": pred_unity_aa[0, 5],
        "spine2": pred_unity_aa[0, 6],
        "l_Ankle": pred_unity_aa[0, 7],
        "r_Ankle": pred_unity_aa[0, 8],
        "spine3": pred_unity_aa[0, 9],
        "l_Foot": pred_unity_aa[0, 10],
        "r_Foot": pred_unity_aa[0, 11],
        "neck": pred_unity_aa[0, 12],
        "l_Collar": pred_unity_aa[0, 13],
        "r_Collar": pred_unity_aa[0, 14],
        "head": pred_unity_aa[0, 15],
        "l_Shoulder": pred_unity_aa[0, 16],
        "r_Shoulder": pred_unity_aa[0, 17],
        "l_Elbow": pred_unity_aa[0, 18],
        "r_Elbow": pred_unity_aa[0, 19],
        "l_Wrist": pred_unity_aa[0, 20],
        "r_Wrist": pred_unity_aa[0, 21]
    }
}

srcJntList = {"angles": {
        "pelvis": pred_unity_aa[0, 0],
        "l_Hip": pred_unity_aa[0, 1],
        "r_Hip": pred_unity_aa[0, 2],
        "spine1": pred_unity_aa[0, 3],
        "l_Knee": pred_unity_aa[0, 4],
        "r_Knee": pred_unity_aa[0, 5],
        "spine2": pred_unity_aa[0, 6],
        "l_Ankle": pred_unity_aa[0, 7],
        "r_Ankle": pred_unity_aa[0, 8],
        "spine3": pred_unity_aa[0, 9],
        "l_Foot": pred_unity_aa[0, 10],
        "r_Foot": pred_unity_aa[0, 11],
        "neck": pred_unity_aa[0, 12],
        "l_Collar": pred_unity_aa[0, 13],
        "r_Collar": pred_unity_aa[0, 14],
        "head": pred_unity_aa[0, 15],
        "l_Shoulder": pred_unity_aa[0, 16],
        "r_Shoulder": pred_unity_aa[0, 17],
        "l_Elbow": pred_unity_aa[0, 18],
        "r_Elbow": pred_unity_aa[0, 19],
        "l_Wrist": pred_unity_aa[0, 20],
        "r_Wrist": pred_unity_aa[0, 21]
    }
}
OMR_instance = OMR(targetList=targetList, endEffectorList=endEffectorList, joint_hierarchy=SMPL_HIERARCHY, joint_data=joint_data, srcJoint_data=srcJntList)
updated_joint_angles_list = []
updated_trans_list = []
for frame_idx in range(1,num_frames):
    joint_data["targets"] = {
        "l_Foot" : pred_unity_pos[frame_idx, 10],
        "r_Foot" : pred_unity_pos[frame_idx, 11]
    }
    joint_data["positions"] = {
        joint_name: fixed_unity_pos[frame_idx, joint_idx] 
        for joint_idx, joint_name in enumerate(joint_names)
    }
    joint_data["angles"] = {
        joint_name: pred_unity_aa[frame_idx, joint_idx] 
        for joint_idx, joint_name in enumerate(joint_names)
    }
    srcJntList["angles"] = {
        joint_name: pred_unity_aa[frame_idx, joint_idx] 
        for joint_idx, joint_name in enumerate(joint_names)
    }
    OMR_instance.joint_data = joint_data
    OMR_instance.srcJoint_data = srcJntList
    start = time.time()
    OMR_instance.ikSolver(frame_idx)
    end = time.time()
    updated_joint_angles = torch.stack([torch.tensor(value, device=device)for value in OMR_instance.joint_data["angles"].values()])

    updated_joint_angles_list.append(updated_joint_angles)

    updated_trans = torch.tensor(list(OMR_instance.joint_data["positions"]["pelvis"]))
    updated_trans_list.append(updated_trans)

updated_joint_angles_tensor = torch.stack(updated_joint_angles_list)
updated_trans_tensor = torch.stack(updated_trans_list)

omr_rmat = aa2matrot(updated_joint_angles_tensor.reshape(-1, 3)).reshape(-1, 22, 3, 3)
omr_rmat = omr_rmat.detach().cpu()
updated_trans_tensor = updated_trans_tensor.detach().cpu()
updated_trans_tensor /= 100.0

if True:
    save_to = os.path.join(generate_output_dir , generate_output_name)
    os.makedirs(os.path.dirname(save_to), exist_ok=True)

    if save_unity_npz is True:
            np.savez_compressed(save_to,
                    
                    mocap_framerate = [60],
                    
                    gt_unity_trans = pred_unity_trans.reshape(num_frames, -1),
                    gt_unity_pose_rmat = pred_unity_rotmat.reshape(num_frames, -1),
                    unity_trans = updated_trans_tensor.reshape(num_frames-1, -1), 
                    unity_pose_rmat = omr_rmat.reshape(num_frames-1, -1), #length
                    
            )
    print("Done.")