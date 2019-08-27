#pragma once
#include <glad/glad.h>

#include <opencv2/core.hpp>

#include <iostream>
#include <vector>

#include "shader.h"

using namespace std;
using namespace cv;

class Background
{
private:
	unsigned int VAO, VBO, EBO;
	shared_ptr<Shader> shader_ptr_;

public:
	Background();
	~Background();

	void Draw(Mat &frame);
};
