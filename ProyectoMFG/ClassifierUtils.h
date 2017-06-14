#pragma once
#include "FaceDetectors.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class ClassiFierUtils
{
public:
	static cv::Mat norm_0_255(cv::InputArray _src);
	static void read_csv(const string& filename, const string& dst, char separator = ',', int size = 1500);
	static void loadImgs(const string& src, vector<cv::Mat>& images, int size);
	static void loadLabels(const string& src, vector<int>& labels, int size);
};