#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader.h"

Shader::Shader(boost::filesystem::path vs_path, boost::filesystem::path fs_path) {
	auto ReadFileAt = [](boost::filesystem::path path) -> std::string {
		std::ifstream ifs(path.c_str());
		std::string res((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
		return res;
	};
	auto vs_source = ReadFileAt(vs_path);
	auto fs_source = ReadFileAt(fs_path);
	std::cout << vs_source << std::endl;
	auto vs_id = Compile(GL_VERTEX_SHADER, vs_source, vs_path);
	auto fs_id = Compile(GL_FRAGMENT_SHADER, fs_source, fs_path);
	id = Link(vs_id, fs_id);
}

Shader::Shader(std::string vs_source, std::string fs_source) {
	auto vs_id = Compile(GL_VERTEX_SHADER, vs_source, "");
	auto fs_id = Compile(GL_FRAGMENT_SHADER, fs_source, "");
	id = Link(vs_id, fs_id);
}

uint32_t Shader::Compile(uint32_t type, const std::string &source, boost::filesystem::path path) {
	uint32_t shader_id = glCreateShader(type);
	const char *temp = source.c_str();
	glShaderSource(shader_id, 1, &temp, nullptr);
	glCompileShader(shader_id);
	int success;
	glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
	if (success) return shader_id;
	int length;
	glGetShaderiv(shader_id, GL_INFO_LOG_LENGTH, &length);
	char *log = new char[length];
	glGetShaderInfoLog(shader_id, length, nullptr, log);
	std::string log_str = log;
	delete[] log;
}

uint32_t Shader::Link(uint32_t vs_id, uint32_t fs_id) {
	uint32_t program_id = glCreateProgram();

	glAttachShader(program_id, vs_id);
	glAttachShader(program_id, fs_id);

	glLinkProgram(program_id);

	glDeleteShader(vs_id);
	glDeleteShader(fs_id);

	int success;
	glGetProgramiv(program_id, GL_LINK_STATUS, &success);
	if (success) return program_id;
	int length;
	glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &length);
	char *log = new char[length];
	glGetProgramInfoLog(program_id, length, nullptr, log);
	std::string log_str = log;
	delete[] log;
}

void Shader::Use() const {
	glUseProgram(id);
}

template <>
void Shader::SetUniform<glm::vec3>(const std::string &identifier, const glm::vec3 &value) const {
	auto location = glGetUniformLocation(id, identifier.c_str());
	if (location < 0) return;
	glUniform3fv(location, 1, value_ptr(value));
}

template <>
void Shader::SetUniform<glm::mat4>(const std::string &identifier, const glm::mat4 &value) const {
	auto location = glGetUniformLocation(id, identifier.c_str());
	if (location < 0) return;
	glUniformMatrix4fv(location, 1, GL_FALSE, value_ptr(value));
}

template <>
void Shader::SetUniform<int32_t>(const std::string &identifier, const int32_t &value) const {
	auto location = glGetUniformLocation(id, identifier.c_str());
	if (location < 0) return ;
	glUniform1i(location, value);
}

template <>
void Shader::SetUniform<float>(const std::string &identifier, const float &value) const {
	auto location = glGetUniformLocation(id, identifier.c_str());
	if (location < 0) return;
	glUniform1f(location, value);
}

template <>
void Shader::SetUniform<std::vector<glm::mat4>>(const std::string &identifier, const std::vector<glm::mat4> &value) const {
	auto location = glGetUniformLocation(id, identifier.c_str());
	if (location < 0) return;
	glUniformMatrix4fv(location, value.size(), GL_FALSE, value_ptr(value[0]));
}
