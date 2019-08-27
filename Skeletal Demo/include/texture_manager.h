#pragma once

#include <boost/filesystem.hpp>

class TextureManager {
public:
	static uint32_t LoadTexture(boost::filesystem::path);
};
