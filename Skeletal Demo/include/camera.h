#pragma once
#include <opencv2\core.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\calib3d.hpp>

#include <glm\glm.hpp>

#include "config.h"

using namespace std;
using namespace glm;
using namespace cv;
using namespace cv::aruco;

class Camera
{
private:
	int fwidth_;//帧宽
	int fheight_;//帧高

	Mat camera_matrix_;//内参
	Mat dist_coeffs_;//畸变系数

	Mat view_matrix_;
	Mat projection_matrix_;

	Ptr<Dictionary> dictionary_;//标记字典

	void setProjection();
public:
	Camera(shared_ptr<Config> config_ptr);
	~Camera();

	bool marker_based_compute(Mat &frame);

	int getWidth();
	int getHeight();

	mat4 get_view_matrix();
	mat4 get_projection_matrix();
};
