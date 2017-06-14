#pragma once
#include "opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class IClassifier {

public:
	virtual ~IClassifier(){}

	virtual bool load(const string& genderModel, const string& ageModel) = 0;
	virtual void trainModel(const string& genderSrc, const string& ageSrc) = 0;
	virtual vector<string> classiffy(cv::Mat face) = 0;
	virtual bool usingMajority() = 0;
	virtual bool usingWeight() = 0;
};