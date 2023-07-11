# FCCF-PCR
Feature-Consistent Coplane-Pair Correspondence- and Fusion-Based Point Cloud Registration (2023) by  Kuo-Liang Chung, Pei-Hsuan Hsieh, and Chia-Chi Hsu.
## Usage
Compiling by following code in the project directory
```
mkdir build; cd build
cmake ..; make;
```
then we can get transformation matrix of source and target point cloud by
```
./FCCF {PATH_TO_SRC_CLOUD} {PATH_TO_TAR_CLOUD} {Voxel_Grid_Size}
```
## Testing enviroment
* Windows Subsystem Linux(Ubuntu 20.04.6 LTS)
* ISO C++ 14

## Acknowledgement
Our work and implementations are inspired and based on the following projects: 
```
【1】李建微, 占家旺. 三维点云配准方法研究进展[J]. 中国图象图形学报, 2022, 27(02): 349-367. 
【2】Jianwei Li, Jiawang Zhan, Ting Zhou, Virgílio A. Bento, Qianfeng Wang. 
Point Cloud Registration and Localization Based on Voxel Plane Features [J]. 
ISPRS Journal of Photogrammetry and Remote Sensing.（expected to be published in this journal）
```
We appreciate the authors for sharing their codes.
## Contact
Please email me if you have any questions!  
Chia-Chi Hsu : m11115040@mail.ntust.edu.tw
