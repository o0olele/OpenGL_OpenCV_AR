#pragma once

#include <map>
#include <string>

class Namer {
public:
	Namer();
	int Name(const std::string &name);
	int total() const;
	std::map<std::string, int> &map();
	void Clear();

private:
	std::map<std::string, int> map_;
	int total_;
};

