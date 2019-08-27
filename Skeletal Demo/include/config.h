#pragma once

#include <iostream>
#include <opencv2\core.hpp>

using namespace cv;
using namespace std;

class Config
{
private:
	FileStorage file_;

public:
	Config(string path);
	~Config();

	template< typename T >
	void get(const char* key, T &value)
	{
		this->file_[key] >> value;
	}
};
