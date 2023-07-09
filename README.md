# FCCF-PCR
code for paper "Feature-Consistent Coplane-Pair Correspondence- and Fusion-Based Point Cloud Registration"
## Usage
Compiling by following code in the project directory
```
mkdir build; cd build
cmake ..; make;
```
then we can get transformation matrix of source and target point cloud by
```
ours {PATH_TO_SRC_CLOUD} {PATH_TO_TAR_CLOUD}
```
## Testing enviroment
* Windows Subsystem Linux(Ubuntu 20.04.6 LTS)
* ISO C++ 14

## Reference
This code is modified from the source code provided by the following paper : 
```
【1】李建微, 占家旺. 三维点云配准方法研究进展[J]. 中国图象图形学报, 2022, 27(02): 349-367. 
【2】Jianwei Li, Jiawang Zhan, Ting Zhou, Virgílio A. Bento, Qianfeng Wang. 
Point Cloud Registration and Localization Based on Voxel Plane Features [J]. 
ISPRS Journal of Photogrammetry and Remote Sensing.（expected to be published in this journal）
```
