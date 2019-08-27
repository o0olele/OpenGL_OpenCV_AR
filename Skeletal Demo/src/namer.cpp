#include "namer.h"

Namer::Namer() {
	Clear();
}

void Namer::Clear() {
	map_.clear();
	total_ = 0;
}

int Namer::Name(const std::string &name) {
	if (map_.count(name))
		return map_[name];
	return map_[name] = total_++;
}

int Namer::total() const {
	return total_;
}

std::map<std::string, int> &Namer::map() {
	return map_;
}
