cmake_minimum_required(VERSION 3.2)

project(Demo)

set(CMAKE_BUILD_TYPE "RELEASE")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# 设置编译后库文件目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
# 设置可执行文件目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

find_package( glfw3 3.3 REQUIRED PATHS ~/project/libglfw/lib/cmake)
find_package( OpenCV REQUIRED PATHS ~/project/libcv/lib/cmake)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加编译可执行文件 
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
add_executable(Demo ${SRC})

target_link_libraries(Demo glfw ${OpenCV_LIBS})
