#include "config.h"


Config::Config(string path)
{
	this->file_.open(path, FileStorage::READ);

	if (!this->file_.isOpened())
	{
		cerr << "file " << path << " open failed!" << endl;
		return;
	}
}

Config::~Config()
{
	if (this->file_.isOpened())
		this->file_.release();
}
