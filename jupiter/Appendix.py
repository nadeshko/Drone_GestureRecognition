# %% md
# Installing OpenPose

# %%
import os
from os.path import exists, join, basename, splitext

git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]

if 1 or not exists(project_name):
    !rm -rf openpose
    # see: https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/949
    print("install new CMake becaue of CUDA10")
    cmake_version = 'cmake-3.20.2-linux-x86_64.tar.gz'
    if not exists(cmake_version):
        !wget -q 'https://cmake.org/files/v3.20/{cmake_version}'
    !tar xfz {cmake_version} --strip-components=1 -C /usr/local

    print("clone openpose")
    !git clone -q --depth 1 $git_repo_url
    print("install system dependencies")
    !apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev
        libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev
        liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev
    print("build openpose")
    !cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`


# %%
# @title Install Body 25B Model
!mkdir /content/openpose/models/pose/body_25b
!wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/1_25BBkg/body_25b/pose_iter_XXXXXX.caffemodel \
    -O /content/openpose/models/pose/body_25b/pose_iter_XXXXXX.caffemodel
!wget https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose_train/master/\
    experimental_models/1_25BBkg/body_25b/pose_deploy.prototxt \
    -O /content/openpose/models/pose/body_25b/pose_deploy.prototxt

# %% md
# OpenPose on Videos
# in content
!mkdir /content/Actions/OpenPose

# %%
# OpenPose on a video under different directories
import os
from os.path import exists, join, basename, splitext

h, w = 480, 720

vid_path = "/content/Actions"
class_names = os.listdir(vid_path)
num_class = len(class_names)

openpose_out = '/content/Actions/OpenPose'
openpose_json = '/content/OP_Results/json'

# save json files and video into another directory
for idx, cname in enumerate(class_names):
    if idx != 4:
        class_path = os.path.join(vid_path, cname)
        video_names = sorted(os.listdir(class_path))
        for vname in video_names:
            video_path = os.path.join(class_path, vname)
            out_vidpath = os.path.join(openpose_out, cname, vname)
            json_vidpath = os.path.join(openpose_json, cname, os.path.splitext(vname)[0])
            if not os.path.exists(json_vidpath):
                os.makedirs(json_vidpath)

            if not os.path.exists(os.path.join(openpose_out, cname)):
                os.makedirs(os.path.join(openpose_out, cname))

            if not os.path.exists(out_vidpath):
                !cd /content/openpose && ./build/examples/openpose/openpose.bin \
                --net_resolution -1x{h} --scale_number 4 --scale_gap 0.25 \
                --video {video_path} \
                --number_people_max 1 \
                --write_json {json_vidpath} \
                --display 0 \
                --write_video {out_vidpath}

