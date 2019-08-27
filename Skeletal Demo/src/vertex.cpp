#include <algorithm>

#include "vertex.h"

Vertex::Vertex() {
	std::fill(bone_ids, bone_ids + kMaxBonesPerVertex, 0);
	std::fill(bone_weights, bone_weights + kMaxBonesPerVertex, 0);
}

void Vertex::AddBone(int id, float weight) {
	int i;
	for (i = 0; i < kMaxBonesPerVertex; i++) if (bone_weights[i] == 0) break;
	if (i >= kMaxBonesPerVertex) return;
	bone_weights[i] = weight;
	bone_ids[i] = id;
}
