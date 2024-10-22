开源之夏——基于Mindspore+香橙派的具身智能应用开发
================
本仓库包含代码如下：
1. arm_control 六自由度机械臂电机控制代码
2. arm_ws 六自由度机械手臂基于ROS的描述文件代码（各个link的mesh由IGS模型中手动选取生成的简化模型）
3. init.ipynb 通过调用mindnlp 的ChatGM4接口判断用户是否要求机械臂抓取物体
4. detection_tracking 其中detection网络的修改自["Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (ECCV2024)](https://github.com/IDEA-Research/GroundingDINO), tracking网络的修改自["FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects" (CVPR 2024)](https://github.com/NVlabs/FoundationPose), 也包含基于pyk4a的相机相关驱动代码
5. graspnet_mindspore 基于Mindspore的平行爪位姿生成网络代码，原工作来自["GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020)](https://github.com/graspnet/graspnet-baseline)（working）
6. Detection&tracking的效果展示(done)，最终的项目展示视频(todo)
