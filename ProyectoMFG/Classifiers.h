#pragma once
#include "IClassifier.h"


#include "opencv2/core/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>


#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>


using namespace cv;
using namespace std;
using namespace cv::ml;

using namespace std;

class BasicGenderClassifier : public IClassifier
{
public:
	BasicGenderClassifier(bool Majority, bool Weighted, cv::Ptr<cv::face::FaceRecognizer> faceRecognizer){
		majority = Majority;
		weighted = Weighted;
		model = faceRecognizer;
	}
	virtual bool load(const string& genderModel, const string& ageModel);
	virtual void trainModel(const string& genderSrc, const string& ageSrc);
	virtual vector<string> classiffy(cv::Mat face);
	virtual bool usingMajority();
	virtual bool usingWeight();
private:
	void trainBasicModel(const string& srcImgs, const string& srcLabels, int size);
	cv::Ptr<cv::face::FaceRecognizer> model;
	bool majority;
	bool weighted;
};


#include "PhogFeatures.h"
class PhogSVMClassifier : public IClassifier
{
public:
	PhogSVMClassifier(bool Majority, bool Weighted){
		majority = Majority;
		weighted = Weighted;
	}
	virtual bool load(const string& genderModel, const string& ageModel);
	virtual void trainModel(const string& genderSrc, const string& ageSrc);
	virtual vector<string> classiffy(cv::Mat face);
	virtual bool usingMajority();
	virtual bool usingWeight();
private:
	bool majority;
	bool weighted;
	void trainModelSVM(Ptr <SVM> svmClassifier, const string& srcVectors, double c, float gamma);
	Ptr <SVM> svmGender;
	Ptr <SVM> svmGender2;
	Ptr <SVM> svmAge;
	Ptr <SVM> svmAge2;
};


