# TSCNet
This project provides the code and results for 'Texture-Semantic Collaboration Network for ORSI Salient Object Detection', IEEE TCAS-II, 2024. [IEEE link](https://ieeexplore.ieee.org/document/10319772) [Homepage](https://mathlee.github.io/)

 
# Network Architecture
   <div align=center>
   <img src="https://github.com/MathLee/TSCNet/blob/main/images/TSCNet.png">
   </div>
   
   
# Requirements
   python 3.8 + pytorch 1.9.0
   

# Saliency maps
   We provide saliency maps saved using two different functions (imageio.imsave and cv2.imwrite) on ORSSD, EORSSD, and ORSI-4199 datasets.
   
   Using "imageio.imsave(res_save_path+name, res)" to save saliency maps in "./models/[saliencymaps_imageio.zip](https://pan.baidu.com/s/1ytlnUknWbJFpC1hFQDniAg)"(code: 6sr5), termed **TSCNet_imageio** in Table I (reported in our paper).
   
   Using "cv2.imwrite(save_path+name, res*256)" to save saliency maps in "./models/saliencymaps_cv2.zip", termed **TSCNet_cv2** in Table I.

      
   ![Image](https://github.com/MathLee/TSCNet/blob/main/images/table.png)
   
# Training

We use data_aug.m for data augmentation.

Download [VGG weight](https://pan.baidu.com/s/10IrazQ8KuxTOx9YJJHi8mg) (code: ipbb), and put it in './model/'.

Download [ViT weight](https://pan.baidu.com/s/1RARIt0EHSOLbng7vEulLLA) (code: nd45), and put it in './network/'.

Run train_TSCNet.py.


# Pre-trained model and testing
Download the following pre-trained model, and modify paths of pre-trained model and datasets, then run test_TSCNet.py.

[ORSSD](https://pan.baidu.com/s/1-KD5Ti2W2wgGIAZPFnWp3g) (code: t6it)

[EORSSD](https://pan.baidu.com/s/1JK8LmCWiFD9E-UxNNJM7ew) (code: f9dv)

[ORSI-4199](https://pan.baidu.com/s/1qpmqL6aZRTP6RTPPhMgV0w) (code: jcm8)

   
# Evaluation Tool
   You can use the [evaluation tool (MATLAB version)](https://github.com/MathLee/MatlabEvaluationTools) to evaluate the above saliency maps.


# [ORSI-SOD_Summary](https://github.com/MathLee/ORSI-SOD_Summary)
   
# Citation
        @ARTICLE{Li_2024_TSCNet,
                author = {Gongyang Li and Zhen Bai and Zhi Liu},
                title = {Texture-Semantic Collaboration Network for ORSI Salient Object Detection},
                journal = {IEEE Transactions on Circuits and Systems II: Express Briefs},
                volume= {71},
                number={4},
                pages={2464-2468},
                year={2024},
                month={Apr.},
                }
                
                
If you encounter any problems with the code, want to report bugs, etc.

Please contact me at lllmiemie@163.com or ligongyang@shu.edu.cn.
