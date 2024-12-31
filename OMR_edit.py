import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from utils.utils_transform import matrot2sixd, sixd2matrot, sixd2aa, aa2matrot, matrot2aa
from utils import smpl_fk_fixed

device = "cuda"
fk_engine = smpl_fk_fixed.smpl_fk_fixed(device=device)
joint2index = {
        'pelvis': 0,
        'l_Hip': 1,
        'r_Hip': 2,
        'spine1': 3,
        'l_Knee': 4,
        'r_Knee': 5,
        'spine2': 6,
        'l_Ankle': 7,
        'r_Ankle': 8,
        'spine3': 9,
        'l_Foot': 10,
        'r_Foot': 11,
        'neck': 12,
        'l_Collar': 13,
        'r_Collar': 14,
        'head': 15,
        'l_Shoulder': 16,
        'r_Shoulder': 17,
        'l_Elbow': 18,
        'r_Elbow': 19,
        'l_Wrist': 20,
        'r_Wrist': 21
    }
class OMR:
    def __init__(self, targetList, endEffectorList, joint_hierarchy, joint_data, srcJoint_data):
        self.targetList = targetList
        self.endEffectorList = endEffectorList
        self.joint_hierarchy = joint_hierarchy
        self.joint_data = joint_data
        self.jntList, self.effectiveJntDic = self.getJntListAndEffectiveJntDic(endEffectorList)
        #self.srcJntList, self.srcEffectiveJntDic = self.getJntListAndEffectiveJntDic(srcEndEffectorList)
        self.srcJoint_data = srcJoint_data
        self.PI = np.pi
        self.prevTargetPosList = None
        self.prevJntAngleList = None
        self.prevSrcJntAngleList = None
    
    def parentJnts(self, endEffector):
        parentJntList = []
        current = endEffector
        while current in self.joint_hierarchy:
            parent = self.joint_hierarchy[current]
            if parent is None:
                break
            parentJntList.append(parent)
            current = parent
        return parentJntList

    def getJntListAndEffectiveJntDic(self, endEffectorList):
        jntList = []
        effectiveJntDic = {}
        for endEffector in endEffectorList:
            effectiveJntDic[endEffector] = self.parentJnts(endEffector)
            jntList.extend(effectiveJntDic[endEffector])
        jntList = list(dict.fromkeys(jntList))
        return jntList, effectiveJntDic

    def clampMag(self, posDisplace, Dmax):
        if (sum(posDisplace*posDisplace)**0.5 <= Dmax ) :
            result = posDisplace
        else :
            result = Dmax*(posDisplace/(sum(posDisplace*posDisplace)**0.5))            
        return result

    def getJacobian(self, jntPosList, jntLocalAxesMat, endEffectorPosList):
        J = np.zeros((len(self.endEffectorList)*3, len(self.jntList)*3))
        
        for i in range(len(self.endEffectorList)): # 3
            for j in range(len(self.jntList)): # 10
                if self.jntList[j] in self.effectiveJntDic[self.endEffectorList[i]]:
                    for k in range(3) :
                        element = np.cross(jntLocalAxesMat[j][k], (endEffectorPosList[i] - jntPosList[j]))
                        J[3*i+0][3*j+k] = element[0]
                        J[3*i+1][3*j+k] = element[1]
                        J[3*i+2][3*j+k] = element[2]
                else :
                    for k in range(3):
                        J[3*i+0][3*j+k] = 0
                        J[3*i+1][3*j+k] = 0
                        J[3*i+2][3*j+k] = 0  
        return J
    def getDisplaceError(self, targetPosList, endEffectorPosList, D=0.0, Dmax=2.0):
        e = np.zeros((len(self.endEffectorList)*3))
        for i in range(len(self.endEffectorList)):
            posDisplace = targetPosList[i] - endEffectorPosList[i]
            posDisplace = self.clampMag(posDisplace, Dmax)
            e[3*i + 0] = posDisplace[0]
            e[3*i + 1] = posDisplace[1]
            e[3*i + 2] = posDisplace[2]
        return e        

    def DampedLeastSquare(self, Jacobian, D=0.0, Dmax=2.0, dampingConstant=3.0):                
        J = Jacobian
        temp = np.dot(J,J.T)+(dampingConstant*dampingConstant)*np.identity(len(self.endEffectorList)*3,float)
        J_inverse = np.dot(J.T, np.linalg.inv(temp))
        return J_inverse  
    
    
    def ikSolver(self, currentTime):
        targetPosMat = np.zeros((len(self.targetList), 3))
        endEffectorPosMat = np.zeros((len(self.endEffectorList), 3))
        jntPosMat = np.zeros((len(self.jntList), 3))
        jntAngleMat = np.zeros((len(self.jntList), 3))
        ## targetPosMat : default target position
        ## endEffectorPosMat : female avatar end effector position
        for i in range(len(self.targetList)):
            target_value = self.joint_data["targets"][self.targetList[i]]
            if isinstance(target_value, torch.Tensor):
                target_value = target_value.cpu().numpy()
            targetPosMat[i] = target_value

            end_effector_value = self.joint_data["positions"][self.endEffectorList[i]]
            if isinstance(end_effector_value, torch.Tensor):
                end_effector_value = end_effector_value.cpu().numpy()
            endEffectorPosMat[i] = end_effector_value
        ## jntPosMat, jntAngleMat = 11x3
        for i in range(len(self.jntList)):
            joint_pos_value = self.joint_data["positions"][self.jntList[i]]
            if isinstance(joint_pos_value, torch.Tensor):
                joint_pos_value = joint_pos_value.cpu().numpy()
            jntPosMat[i] = joint_pos_value

            joint_angle_value = self.joint_data["angles"][self.jntList[i]]
            if isinstance(joint_angle_value, torch.Tensor):
                joint_angle_value = joint_angle_value.cpu().numpy()
            jntAngleMat[i] = joint_angle_value
        # if currentTime >= 140:
        #     print("before", jntAngleMat[1])
        jntLocalAxesMat = aa2matrot(torch.tensor(jntAngleMat).reshape(-1,3)).reshape(-1,3,3).detach().cpu().numpy()    

        J = self.getJacobian(jntPosMat, jntLocalAxesMat, endEffectorPosMat)
        J_inverse = self.DampedLeastSquare(J)
        # print(np.linalg.matrix_rank(J))
        #J_inverse = np.linalg.pinv(J)

        if currentTime == 1:
            self.prevTargetPosList = targetPosMat
            displaceTargetPos = self.getDisplaceError(targetPosMat, endEffectorPosMat)
        else:
            displaceTargetPos = self.getDisplaceError(self.prevTargetPosList, targetPosMat)
            jntAngles = np.dot(J_inverse, displaceTargetPos)
            jntAngles_deg = jntAngles.reshape(-1, 3)
            for i in range(len(self.jntList)): 
                current_angle = jntAngleMat[i]  # 기존 각도
                new_angle = jntAngles_deg[i]  # 추가로 더해야 할 각도 변화
                updated_angle = current_angle + new_angle  # 최종 각도 계산
                self.joint_data["angles"][self.jntList[i]] = updated_angle
            # if currentTime >= 140:
            #     print("after",self.joint_data["angles"]["l_Shoulder"])
            # if currentTime >= 140:
            #     print("before",endEffectorPosMat[0])
            ## Need FK
            fullbody_angle = torch.stack([torch.tensor(value) for value in self.joint_data["angles"].values()]).float().to(device)
            trans = torch.tensor(self.joint_data["positions"]["pelvis"]).unsqueeze(0).float().to(device)
            fullbody_rmat = aa2matrot(fullbody_angle).unsqueeze(0)
            (position,_),(_,_) = fk_engine.fk_batch((trans/100), fullbody_rmat)
            position = 100*position.squeeze(0)
            # if currentTime >= 140:
            #     print("after",position[20])
            
            for joint in self.jntList:
                index = joint2index.get(joint)
                self.joint_data["positions"][joint] = position[index].detach().cpu().numpy()

            tempTargetPosMat = np.zeros((len(self.targetList), 3))

            for i in range(len(self.endEffectorList)):
                tempTargetPosMat[i] = self.joint_data["positions"][self.endEffectorList[i]]
            e = self.getDisplaceError(targetPosMat, tempTargetPosMat)
            displaceTargetPos += e

            # for i in range(len(self.jntList)):
            #     jntAngles_deg_neg = -jntAngles_deg[i]
            #     self.joint_data["angles"][self.jntList[i]] = jntAngles_deg_neg

            self.prevTargetPosList = targetPosMat
            
        srcJntAngles = np.zeros((len(self.jntList) * 3))
        for i, joint in enumerate(self.jntList):
            rot = self.srcJoint_data["angles"][joint]
            srcJntAngles[3 * i] = rot[0]
            srcJntAngles[3 * i + 1] = rot[1]
            srcJntAngles[3 * i + 2] = rot[2]

        if self.prevSrcJntAngleList is None:
            displaceSrcJntAngle = np.zeros(len(self.jntList) * 3)
        else:
            displaceSrcJntAngle = srcJntAngles - self.prevSrcJntAngleList

        tempMat = np.identity(len(self.jntList) * 3) - np.dot(J_inverse, J)
        secondaryJntAngles = np.dot(tempMat, displaceSrcJntAngle)
        jntAngles = np.dot(J_inverse, displaceTargetPos) + secondaryJntAngles

        self.prevJntAngleList = jntAngles
        self.prevSrcJntAngleList = srcJntAngles
        jntAngles = jntAngles.reshape(-1, 3)

        for i in range(len(self.jntList)):
            current_angle = self.joint_data["angles"][self.jntList[i]]
            new_angle = jntAngles[i]
            updated_angle = current_angle + new_angle
            self.joint_data["angles"][self.jntList[i]] = updated_angle
        
            #print(self.joint_hierarchy.items())
        

