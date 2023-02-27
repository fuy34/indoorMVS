# IndoorMVS: Indoor Priors Inspired Multi-view Stereo Network

This is a PyTorch implementation of the indoor MVS method proposed in my [Ph.D. dissertation](https://drive.google.com/file/d/1gvmFLj1Q37N783XiKvvf969EV0co3hr6/view) (Chapter 4). We developed 
a new indoor 3D plan instance dataset and proposed a TSDF-based MVS model that can simultaneously estimate the 3D geometry,
semantics, and major plane instances from posed images. 

Please feel free to contact [Fengting Yang](http://personal.psu.edu/fuy34/) (fuy34bkup@gmail.com) if you have any questions.

## News
We have proposed another powerful indoor MVS method called [PlanarRecon](https://github.com/neu-vi/PlanarRecon). Please feel free to check it out. 

## Prerequisites
The code is developed and tested with
 * Python 3.6
 * PyTorch 1.6.0 (w/ Cuda 10.2)
 * Pytorch-Lightning 1.4.9

You may need to install some other packages, e.g., ```trimesh```, ```opencv```, ```pillow```, etc.  

## Data Preperation
First, download [Scannet](http://www.scan-net.org/) dataset and extract it with the official instruction to ```<SCANNET_PATH>```. 

Next,  download our ground truth 3D plane instance from [Here](https://drive.google.com/file/d/1cFCAGCWMd5ciornZQKkr21OpaaO9ajdx/view?usp=sharing) and save it at ```<GT_INS_PATH>```. This ground truth is generated with an adopted plane 
fitting algorithm from [planeRCNN](https://github.com/NVlabs/planercnn).  If you wish to generate the ground truth by yourself, 
you can first run 
```
python third_part/ScanNet_preprocess/fit_plane/get_gtPlane_segmt.py
```
to obtain the initial plane instances, and then run 
```
python third_part/ScanNet_preprocess/fit_plane/refine_plane_ins.py 
```
to refine the plane instance. Extreme slim and long planes will be removed. Please make sure all the ```path variables``` in the two ```.py```
files are aligned with your local settings. 

Finally, run
```
python prepare_data_tsdf.py --path <SCANNET_PATH> --plane_path <GT_INS_PATH> --path_meta <SAVE_PATH> --i 0 --n 2&
python prepare_data_tsdf.py --path <SCANNET_PATH> --plane_path <GT_INS_PATH> --path_meta <SAVE_PATH> --i 1 --n 2&
```
where ```n``` equals the GPU number you wish to use. The code will run in parallel on multiple GPUs. 

## Training
Once the data is prepared, we should be able to train the model by running the following command. For GPUs with small memory (e.g., Titan XP),
we recommend setting ```MODEL.RESNETS.DEPTH 18``` (ResNet18-FPN), and for ones with larger memory (e.g., RTX 3090 Ti), choosing ```MODEL.RESNETS.DEPTH 50```
(ResNet50-FPN) will lead to better performance.
```
python train.py LOG_DIR <LOG_PATH>  MODEL.RESNETS.DEPTH 18/50 TRAINER.NUM_GPUS 4 RESUME_CKPT <IF_CONTINUE_TRAINING> DATA.NUM_FRAMES_TRAIN 50 TRAINER.PRECISION 16
```
There are a few other hyperparameters in ```config.py``` file. You may need to modify ```DATASETS_TRAIN``` and ```DATASETS_VAL```
according to the location of your training and validation ```.txt``` files. For the others, you can simply use the default values unless you wish to train on different settings.
It may take a few minutes to initialize the voxel indices at the first time you run the algorithm.

## Inference
We provide our pre-trained model (ResNet50-FPN backbone) [here](https://drive.google.com/file/d/1LRk1yl97cLhmHRMquPad5sF4OKOzUSaV/view?usp=sharing). To infer the 3D scene, run  
```
CUDA_VISIBLE_DEVICES=0 python inference.py --model <CKPT_PATH> --scenes <VAL_SCENE_TXT_PTH> --heatmap_thres 0.008
```
The results will be saved in a new folder ```<INFER_PATH>``` in ```<LOG_PATH>```. The folder will include two directories, called ```semseg``` and ```plane_ins```, 
and a few ```.npz``` and ```.ply``` files. The first two directories contain the semantic segmentation and plane segmentation, respectively, and the other
files record the 3D geometry. 


## Evaluation
We evaluate both the 3D reconstruction and semantic estimation accuracy. 

Given the inference results in ```<INFER_PATH>```, for 3D reconstruction evaluation, we can run
```
python evaluate.py --results <INFER_PATH> --scenes <VAL_SCENE_TXT_PTH>
```
Both 2D depth and 3D mesh results will be printed at the end. 

For semantic evaluation, first run 
```
python third_party/Scannet_eval/export_train_mesh_for_evaluation.py --scan_path  <SCANNET_PATH> --scan_list <VAL_SCENE_TXT_PTH> --output_path <SEM_GT_PATH> 
```
to generate the semantic ground truth ```.txt```, and then run 
```
python third_party/Scannet_eval/evaluate_semantic_label.py --pred_path <INFER_PATH>/semseg --gt_path <SEM_GT_PATH> --scan_list <VAL_SCENE_TXT_PTH> 
```
to obtain the semantic evaluation result. 

## Citation
If you find this work helpful, please consider citing my dissertation.
```
@phdthesis{yang2022IndoorMVS,
  title={Geometry Inspired Deep Neural Networks for 3D Reconstruction},
  author={Yang, Fengting},
  school={The Pennsylvania State Univeristy},
  year={2022}
}
```
## Acknowledgement 
Parts of the code are developed from [Atlas](https://github.com/magicleap/Atlas), [PlaneRCNN](https://github.com/NVlabs/planercnn),
[MonoDepthv2](https://github.com/nianticlabs/monodepth2), [HT-LCNN](https://github.com/yanconglin/Deep-Hough-Transform-Line-Priors),
and [HSM](https://github.com/gengshan-y/high-res-stereo).
