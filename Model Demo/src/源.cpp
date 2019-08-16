#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include <opencv2\core.hpp>
#include <opencv2\core\opengl.hpp>
#include <opencv2\core\cuda.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\aruco.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>

//#include "stb_image.h"
#include "shader.h"
#include "camera.h"
#include "model.h"

#include <iostream>

using namespace cv;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 640;
const unsigned int SCR_HEIGHT = 480;

GLuint matricesUniBuffer;
#define MatricesUniBufferSize sizeof(float) * 16 * 3
#define ProjMatrixOffset 0
#define ViewMatrixOffset sizeof(float) * 16
#define ModelMatrixOffset sizeof(float) * 16 * 2
#define MatrixSize sizeof(float) * 16

VideoCapture cap(0);

cv::Mat camera_matrix, dist_coeffs;
cv::Ptr<cv::aruco::Dictionary> dictionary;

std::vector< int > markerIds;
std::vector< std::vector<cv::Point2f> > markerCorners, rejectedCandidates;
cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();
float markerLength = 1.75;

cv::Mat viewMatrix = cv::Mat::zeros(4, 4, CV_32F);
bool is_mark = false;

int minHessian = 400;
std::vector<KeyPoint> keypoints_object, keypoints_scene;
Mat descriptors_object, descriptors_scene;
Mat img_object = imread("src.jpg", IMREAD_GRAYSCALE);
Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);

//初始化已知物体的特征描述
void initKnownDescriptors()
{
	detector->detectAndCompute(img_object, noArray(), keypoints_object, descriptors_object);
}

void findAndComputePose(cv::Mat &img_scene)
{
	//计算当前帧的特征
	std::vector<KeyPoint> keypoints_scene;
	Mat descriptors_scene;
	detector->detectAndCompute(img_scene, noArray(), keypoints_scene, descriptors_scene);

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
	if (good_matches.size() > 8)
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
			obj_corners_3d[i] = Point3f(obj_corners[i].x/640.f-0.5, -obj_corners[i].y/640.f+0.5, 0);
		//相机姿态估计
		cv::Vec3d rvec, tvec;
		cv::solvePnP(obj_corners_3d, scene_corners, camera_matrix, dist_coeffs, rvec, tvec);

		//以下同maker-based 相同
		cv::Mat viewMatrixf = cv::Mat::zeros(4, 4, CV_32F);
		cv::Mat rot;

		Rodrigues(rvec, rot);
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				viewMatrixf.at<float>(row, col) = (float)rot.at<double>(row, col);
			}
			viewMatrixf.at<float>(row, 3) = (float)tvec[row];
		}
		viewMatrixf.at<float>(3, 3) = 1.0f;

		cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
		cvToGl.at<float>(0, 0) = 1.0f;
		cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
		cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
		cvToGl.at<float>(3, 3) = 1.0f;
		viewMatrixf = cvToGl * viewMatrixf;
		cv::transpose(viewMatrixf, viewMatrixf);

		viewMatrix = viewMatrixf;

		is_mark = true;
	}
	else
	{
		is_mark = false;
	}

}

void readCameraPara()
{
	dictionary = cv::aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));

	cv::FileStorage fs("camera.yml", cv::FileStorage::READ);

	fs["camera_matrix"] >> camera_matrix;
	fs["distortion_coefficients"] >> dist_coeffs;

	std::cout << "camera_matrix\n"
		<< camera_matrix << std::endl;
	std::cout << "\ndist coeffs\n"
		<< dist_coeffs << std::endl;
}

void detectArucoMarkers(cv::Mat &image) {
	cv::aruco::detectMarkers(
		image,        // input image
		dictionary,        // type of markers that will be searched for
		markerCorners,    // output vector of marker corners
		markerIds,        // detected marker IDs
		detectorParams,    // algorithm parameters
		rejectedCandidates);

	if (markerIds.size() > 0) {
		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);

		std::vector< cv::Vec3d > rvecs, tvecs;

		cv::aruco::estimatePoseSingleMarkers(
			markerCorners,    // vector of already detected markers corners
			markerLength,    // length of the marker's side
			camera_matrix,     // input 3x3 floating-point instrinsic camera matrix K
			dist_coeffs,       // vector of distortion coefficients of 4, 5, 8 or 12 elements
			rvecs,            // array of output rotation vectors 
			tvecs);            // array of output translation vectors

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Vec3d r = rvecs[i];
			cv::Vec3d t = tvecs[i];

			cv::Mat viewMatrixf = cv::Mat::zeros(4, 4, CV_32F);
			cv::Mat rot;

			Rodrigues(rvecs[i], rot);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{
					viewMatrixf.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				viewMatrixf.at<float>(row, 3) = (float)tvecs[i][row];
			}
			viewMatrixf.at<float>(3, 3) = 1.0f;

			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
			cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
			cvToGl.at<float>(3, 3) = 1.0f;
			viewMatrixf = cvToGl * viewMatrixf;
			cv::transpose(viewMatrixf, viewMatrixf);

			viewMatrix = viewMatrixf;

		}
		is_mark = true;

	}
	else
	{
		is_mark = false;
	}

}

void setModelMatrix() {

	glm::mat4 model;
	model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
	model = glm::scale(model, glm::vec3(0.2, 0.2, 0.2));
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER,
		ModelMatrixOffset, MatrixSize, glm::value_ptr(model));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

static Mat_<float> projMatrix;
void buildProjectionMatrix(float nearp, float farp) {
	projMatrix.create(4, 4); projMatrix.setTo(0);

	float f_x = camera_matrix.at<double>(0, 0);
	float f_y = camera_matrix.at<double>(1, 1);

	float c_x = camera_matrix.at<double>(0, 2);
	float c_y = camera_matrix.at<double>(1, 2);

	projMatrix.at<float>(0, 0) = 2 * f_x / (float)SCR_WIDTH;
	projMatrix.at<float>(1, 1) = 2 * f_y / (float)SCR_HEIGHT;

	projMatrix.at<float>(2, 0) = 1.0f - 2 * c_x / (float)SCR_WIDTH;
	projMatrix.at<float>(2, 1) = 2 * c_y / (float)SCR_HEIGHT - 1.0f;
	projMatrix.at<float>(2, 2) = -(farp + nearp) / (farp - nearp);
	projMatrix.at<float>(2, 3) = -1.0f;

	projMatrix.at<float>(3, 2) = -2.0f*farp*nearp / (farp - nearp);

	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ProjMatrixOffset, MatrixSize, projMatrix.data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

void setCamera(cv::Mat viewMatrix) {
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER, ViewMatrixOffset, MatrixSize, (float*)viewMatrix.data);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

int main()
{
	readCameraPara();
	//initKnownDescriptors();

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	/*********************************背景*************************************/
	Shader texShader("texture.vs", "texture.fs");
	float tex_vertices[] = {
		1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
		1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
	};
	unsigned int indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};
	unsigned int TVBO, TVAO, TEBO;
	glGenVertexArrays(1, &TVAO);
	glGenBuffers(1, &TVBO);
	glGenBuffers(1, &TEBO);

	glBindVertexArray(TVAO);

	glBindBuffer(GL_ARRAY_BUFFER, TVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(tex_vertices), tex_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coord attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	/*********************************模型*************************************/
	glEnable(GL_DEPTH_TEST);

	Model ourModel("D:\\Program Files\\opencv\\proj\\artest\\arcard\\models\\班长符华.pmx");
	Shader ourShader("shader.vs", "shader.fs");

	//
	// Uniform Block
	//
	unsigned int uniformBlockIndex = glGetUniformBlockIndex(ourShader.ID, "Matrices");
	glUniformBlockBinding(ourShader.ID, uniformBlockIndex, 0);

	glGenBuffers(1, &matricesUniBuffer);
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferData(GL_UNIFORM_BUFFER, MatricesUniBufferSize, NULL, GL_DYNAMIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, matricesUniBuffer, 0, MatricesUniBufferSize);
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

	buildProjectionMatrix(0.01f, 1000.0f);

	Mat frame;

	while (!glfwWindowShouldClose(window))
	{
		processInput(window);
		/*********************************背景*************************************/
		unsigned int texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture); 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		cap >> frame;
		
		detectArucoMarkers(frame);
		//findAndComputePose(frame);
		cv::flip(frame, frame, 0);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glBindTexture(GL_TEXTURE_2D, texture);

		texShader.use();

		glBindVertexArray(TVAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		/*******************************模型***************************************/

		glClear(GL_DEPTH_BUFFER_BIT);

		setCamera(viewMatrix);

		setModelMatrix();

		ourShader.use();

		if (is_mark)
		{
			ourModel.Draw(ourShader);
		}
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glDeleteVertexArrays(1, &TVAO);
	glDeleteBuffers(1, &TVBO);
	glDeleteBuffers(1, &TEBO);

	glfwTerminate();
	return 0;
}

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}