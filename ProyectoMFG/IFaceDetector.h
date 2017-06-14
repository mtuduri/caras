#pragma once
//  ****** BASE FACE DETECTOR ******
#include "opencv.hpp"

using namespace std;

class IFaceDetector {

public:
	virtual ~IFaceDetector(){}
	virtual vector<cv::Rect> detectFaces(cv::Mat frame, double scaleFactor, int minNeighbours, int flags, cv::Size minSize) = 0;
	virtual void init(cv::String cascadePath) = 0;
};