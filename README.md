# Motion-Retargeting
AvatarJLM의 사전 훈련된 모델 사용
https://github.com/zxz267/AvatarJLM 에서 다운 후
터미널에서 아래의 명령어 실행.
python generate_VR_motion.py --checkpoint ./results/AvatarJLM/models/AvatarJLM-p1-100k.pth

Motion Retargeting method는 Online-Motion retargeting의 method 사용.(원본 코드는 Maya 라이브러리를 사용. 현재는 numpy, cuda를 사용해 수식 처리함)
원본 코드 : https://github.com/songjaewon/OMR
