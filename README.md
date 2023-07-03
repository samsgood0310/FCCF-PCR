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
