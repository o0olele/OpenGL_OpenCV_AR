#pragma once
#include <opencv2\core.hpp>
#include <opencv2\core\opengl.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>

#include <iostream>

using namespace cv;

class CameraPose
{
public:
	Mat viewMatrix;
	bool using_markerless;
	bool is_mark;

	Mat getCamMatrix();

	void readCamParameters(String );
	void pose_estimate(Mat &in);
	void marker_based(Mat &in);
	void markerless(cv::Mat &in);

	CameraPose(bool, String, String);
	~CameraPose();

private:
	float markerLength = 1.75;
	int minHessian = 400;

	Mat camera_matrix;
	Mat dist_coeffs;

	Ptr<aruco::Dictionary> dictionary;
	Ptr<aruco::DetectorParameters> detectorParams;
	std::vector< int > markerIds;
	std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;
	
	Mat img_object;
	Mat descriptors_object;
	std::vector<KeyPoint> keypoints_object;
	Ptr<cv::xfeatures2d::SURF> detector;

};

Mat CameraPose::getCamMatrix()
{
	return this->camera_matrix;
}

void CameraPose::readCamParameters(String path)
{
	this->dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));

	cv::FileStorage fs(path, cv::FileStorage::READ);

	fs["camera_matrix"] >> this->camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;

	std::cout << "camera_matrix\n"
		<< camera_matrix << std::endl;
	std::cout << "\ndist coeffs\n"
		<< dist_coeffs << std::endl;
}

void CameraPose::pose_estimate(Mat & in)
{
	if (using_markerless)
		this->markerless(in);
	else
		this->marker_based(in);
}

void CameraPose::marker_based(cv::Mat &image) {
	cv::aruco::detectMarkers(image, this->dictionary, markerCorners, markerIds, this->detectorParams, rejectedCandidates);

	if (markerIds.size() > 0) {
		std::vector< cv::Vec3d > rvecs, tvecs;

		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
		cv::aruco::estimatePoseSingleMarkers(markerCorners, this->markerLength, this->camera_matrix, this->dist_coeffs, rvecs, tvecs);

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Mat rot;
			cv::Vec3d r = rvecs[i];
			cv::Vec3d t = tvecs[i];

			this->viewMatrix = cv::Mat::zeros(4, 4, CV_32F);

			Rodrigues(rvecs[i], rot);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{
					this->viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				this->viewMatrix.at<float>(row, 3) = (float)tvecs[i][row];
			}
			this->viewMatrix.at<float>(3, 3) = 1.0f;

			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f;
			cvToGl.at<float>(2, 2) = -1.0f;
			cvToGl.at<float>(3, 3) = 1.0f;
			this->viewMatrix = cvToGl * this->viewMatrix;
			cv::transpose(this->viewMatrix, this->viewMatrix);
		}
		this->is_mark = true;
	}
	else
	{
		this->is_mark = false;
	}

}

void CameraPose::markerless(cv::Mat &img_scene)
{
	//计算当前帧的特征
	std::vector<KeyPoint> keypoints_scene;
	Mat descriptors_scene;
	this->detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

	if (keypoints_scene.size() <= 0)
		return;
	//特征匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;

	matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

	//精简特征
	const float ratio_thresh = 0.75f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	if (good_matches.size() > 20)
	{
		//计算单应性矩阵
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, RANSAC);

		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(0, 0);
		obj_corners[1] = Point2f((float)img_object.cols, 0);
		obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
		obj_corners[3] = Point2f(0, (float)img_object.rows);

		//应用单应性矩阵
		std::vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners, scene_corners, H);

		std::vector<Point3f> obj_corners_3d(4);
		for (size_t i = 0; i < 4; i++)
			obj_corners_3d[i] = Point3f(obj_corners[i].x / 640.f - 0.5, -obj_corners[i].y / 640.f + 0.5, 0);
		//相机姿态估计
		cv::Vec3d rvec, tvec;
		cv::solvePnP(obj_corners_3d, scene_corners, camera_matrix, dist_coeffs, rvec, tvec);

		//以下同maker-based 相同
		viewMatrix = cv::Mat::zeros(4, 4, CV_32F);
		cv::Mat rot;

		Rodrigues(rvec, rot);
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				this->viewMatrix.at<float>(row, col) = (float)rot.at<double>(row, col);
			}
			this->viewMatrix.at<float>(row, 3) = (float)tvec[row];
		}
		this->viewMatrix.at<float>(3, 3) = 1.0f;

		cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
		cvToGl.at<float>(0, 0) = 1.0f;
		cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
		cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
		cvToGl.at<float>(3, 3) = 1.0f;
		this->viewMatrix = cvToGl * this->viewMatrix;
		cv::transpose(this->viewMatrix, this->viewMatrix);

		this->is_mark = true;
	}
	else
	{
		this->is_mark = false;
	}

}

CameraPose::CameraPose(bool use_markerless, String camera_params_file_path, String markerless_srcfile_path="src.jpg")
{
	this->readCamParameters(camera_params_file_path);

	this->using_markerless = use_markerless;
	this->is_mark = false;
	this->viewMatrix = Mat::zeros(4, 4, CV_32F);
	this->detectorParams = aruco::DetectorParameters::create();

	if (this->using_markerless) {
		this->img_object = imread(markerless_srcfile_path, IMREAD_GRAYSCALE);
		this->detector = xfeatures2d::SURF::create(this->minHessian);
		this->detector->detectAndCompute(this->img_object, noArray(), this->keypoints_object, this->descriptors_object);
	}

}

CameraPose::~CameraPose()
{
}