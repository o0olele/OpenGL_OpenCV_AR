# Usage
## OpenCV

~~~shell
# directory
#├── project
#│   ├── libcv
#│   ├── opencv
#│   └── opencv_contrib

# install git
sudo apt-get install git
# source code
# gitee https://gitee.com/mirrors/opencv.git
git clone https://github.com/opencv/opencv.git
# https://gitee.com/mirrors/opencv_contrib.git
git clone https://github.com/opencv/opencv_contrib.git
# prepare
cd opencv
mkdir build && cd build
# cmake params
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=~/project/libcv \
-D OPENCV_EXTRA_MODULES_PATH=~/project/opencv_contrib/modules \
-D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_IPP=ON \
-D WITH_OPENGL=ON -D WITH_EIGEN=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_SHARED_LIBS=OFF ..
# make
make -j8
# install
make install
~~~

## GLFW & GLAD

~~~shell
# dir
#├── project
#│   ├── libglfw
#│   ├── glfw

# download source from https://www.glfw.org/download.html
cd glfw
mkdir build
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/project/libglfw ..
make install
~~~

## GLM

~~~shell
# dir
#├── project
#│   ├── glm

git clone https://github.com/g-truc/glm
cp -r glm/glm ./"Cube Demo Linux"/include

~~~

## Make

~~~shell
cd ./"Cube Demo Linux"
cmake .
make
# before run, you need generate the camera parameters (Camera Calibration) for your own camera.
# see https://blog.csdn.net/qq_33446100/article/details/89192005
# or
# https://docs.opencv.org/4.0.1/da/d13/tutorial_aruco_calibration.html
./Demo

~~~

# Result
![ai](https://img-blog.csdnimg.cn/20210515141040748.png)
