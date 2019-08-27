
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

#include <memory>

#include "background.h"
#include "camera.h"
#include "config.h"
#include "sprite.h"

using namespace cv;
using namespace std;

shared_ptr<Config> config_ptr;
shared_ptr<Camera> camera_ptr;
std::shared_ptr<SpriteModel> sprite_model_ptr;

GLFWwindow *window;

void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void Init()
{
	config_ptr = make_shared<Config>("camera.yml");
	camera_ptr = make_shared<Camera>(config_ptr);

	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	window = glfwCreateWindow(camera_ptr->getWidth(), camera_ptr->getHeight(), "AR", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return;
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

int main()
{
	Init();
	sprite_model_ptr = make_shared<SpriteModel>(".\\models\\sprite\\sprite.fbx");
	bool has_marker = false;
	VideoCapture cap(1);
	shared_ptr<Background> background_ptr = make_shared<Background>();
	int length = 8;
	while (!glfwWindowShouldClose(window))
	{
		static double last_time = glfwGetTime();
		double current_time = glfwGetTime();
		last_time = current_time;

		Mat frame;
		processInput(window);
		/*********************************±³¾°*************************************/
		cap >> frame;
		has_marker = camera_ptr->marker_based_compute(frame);
		cv::flip(frame, frame, 0);
		background_ptr->Draw(frame);

		/*******************************Ä£ÐÍ***************************************/
		glClear(GL_DEPTH_BUFFER_BIT);

		current_time = current_time - int(current_time) / length * length;

		if (has_marker)
			sprite_model_ptr->Draw(0, camera_ptr, current_time);
		

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}