#define STB_IMAGE_IMPLEMENTATION
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#include <opencv2\core.hpp>

#include "camera_pose.h"
#include "shader.h"
#include "model.h"
#include "backmesh.h"

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

CameraPose ourCameraPose(false, "camera.yml");

void setModelMatrix() {

	glm::mat4 model;
	model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0, 0.0, 0.0));
	model = glm::scale(model, glm::vec3(0.1, 0.1, 0.1));
	glBindBuffer(GL_UNIFORM_BUFFER, matricesUniBuffer);
	glBufferSubData(GL_UNIFORM_BUFFER,
		ModelMatrixOffset, MatrixSize, glm::value_ptr(model));
	glBindBuffer(GL_UNIFORM_BUFFER, 0);

}

static Mat_<float> projMatrix;
void buildProjectionMatrix(float nearp, float farp) {
	projMatrix.create(4, 4); projMatrix.setTo(0);

	float f_x = ourCameraPose.getCamMatrix().at<double>(0, 0);
	float f_y = ourCameraPose.getCamMatrix().at<double>(1, 1);

	float c_x = ourCameraPose.getCamMatrix().at<double>(0, 2);
	float c_y = ourCameraPose.getCamMatrix().at<double>(1, 2);

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
	vector<float> tex_vertices = {
		1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
		1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
	};
	vector<unsigned int> indices = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};
	BackMesh ourBackMesh(tex_vertices, indices, &texShader);

	/*********************************模型*************************************/
	glEnable(GL_DEPTH_TEST);

	Model ourModel(".\\models\\班长符华.pmx");
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
		cap >> frame;
		
		ourCameraPose.pose_estimate(frame);
		cv::flip(frame, frame, 0);

		ourBackMesh.Draw(frame);

		/*******************************模型***************************************/

		glClear(GL_DEPTH_BUFFER_BIT);

		setCamera(ourCameraPose.viewMatrix);

		setModelMatrix();

		ourShader.use();

		if (ourCameraPose.is_mark)
		{
			ourModel.Draw(ourShader);
		}
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

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