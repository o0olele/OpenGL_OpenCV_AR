#include "camera.h"

#define MarkerLength 1.75

void Camera::setProjection()
{
	float farp = 100, nearp = 0.1;
	projection_matrix_= Mat::zeros(4, 4, CV_32F);

	float f_x = camera_matrix_.at<double>(0, 0);
	float f_y = camera_matrix_.at<double>(1, 1);

	float c_x = camera_matrix_.at<double>(0, 2);
	float c_y = camera_matrix_.at<double>(1, 2);

	projection_matrix_.at<float>(0, 0) = 2 * f_x / (float)fwidth_;
	projection_matrix_.at<float>(1, 1) = 2 * f_y / (float)fheight_;

	projection_matrix_.at<float>(2, 0) = 1.0f - 2 * c_x / (float)fwidth_;
	projection_matrix_.at<float>(2, 1) = 2 * c_y / (float)fheight_ - 1.0f;
	projection_matrix_.at<float>(2, 2) = -(farp + nearp) / (farp - nearp);
	projection_matrix_.at<float>(2, 3) = -1.0f;

	projection_matrix_.at<float>(3, 2) = -2.0f*farp*nearp / (farp - nearp);
}

Camera::Camera(shared_ptr<Config> config_ptr)
{
	config_ptr->get("image_width", fwidth_);
	config_ptr->get("image_height", fheight_);
	config_ptr->get("camera_matrix", camera_matrix_);
	config_ptr->get("distortion_coefficients", dist_coeffs_);

	dictionary_ = getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));

	setProjection();
}

Camera::~Camera()
{
}

bool Camera::marker_based_compute(Mat &image)
{
	Ptr<DetectorParameters> detectorParams = DetectorParameters::create();

	vector< int > markerIds;
	vector< vector<cv::Point2f> > markerCorners, rejectedCandidates;

	cv::aruco::detectMarkers(image, dictionary_, markerCorners, markerIds, detectorParams, rejectedCandidates);

	if (markerIds.size() > 0) {
		std::vector< cv::Vec3d > rvecs, tvecs;

		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
		cv::aruco::estimatePoseSingleMarkers(markerCorners, MarkerLength, camera_matrix_, dist_coeffs_, rvecs, tvecs);

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Mat rot;
			cv::Vec3d r = rvecs[i];
			cv::Vec3d t = tvecs[i];

			view_matrix_ = cv::Mat::zeros(4, 4, CV_32F);

			Rodrigues(rvecs[i], rot);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{
					view_matrix_.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				view_matrix_.at<float>(row, 3) = (float)tvecs[i][row];
			}
			view_matrix_.at<float>(3, 3) = 1.0f;

			//cv to gl
			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f;
			cvToGl.at<float>(2, 2) = -1.0f;
			cvToGl.at<float>(3, 3) = 1.0f;
			view_matrix_ = cvToGl * view_matrix_;
			cv::transpose(view_matrix_, view_matrix_);
		}
		return true;
	}

	return false;
}

int Camera::getWidth()
{
	return fwidth_;
}

int Camera::getHeight()
{
	return fheight_;
}

mat4 Camera::get_view_matrix()
{
	glm::mat4 temp;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			temp[i][j] = view_matrix_.at<float>(i, j);

	return temp;
}

mat4 Camera::get_projection_matrix()
{
	glm::mat4 temp;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			temp[i][j] = projection_matrix_.at<float>(i, j);

	return temp;
}
