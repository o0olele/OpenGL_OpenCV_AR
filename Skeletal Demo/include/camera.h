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
	int fwidth_;//֡��
	int fheight_;//֡��

	Mat camera_matrix_;//�ڲ�
	Mat dist_coeffs_;//����ϵ��

	Mat view_matrix_;
	Mat projection_matrix_;

	Ptr<Dictionary> dictionary_;//����ֵ�

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
