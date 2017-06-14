#include "FaceDetectors.h"

void OpenCLFaceDetector::init(cv::String cascadePath)
{
	cascade = cv::makePtr<cv::CascadeClassifier>(cascadePath);
	cv::ocl::setUseOpenCL(true);
}

vector<cv::Rect> OpenCLFaceDetector::detectFaces(cv::Mat frame, double scaleFactor = 1.1000000000000001, int minNeighbours = 3, int flags = 0, cv::Size minSize = cv::Size(30, 30))
{
	cv::UMat uframe, uFrameGray;
	vector<cv::Rect>faces;
	frame.copyTo(uframe);
	cv::cvtColor(uframe, uFrameGray, CV_BGR2GRAY);
	cascade->detectMultiScale(uFrameGray, faces, scaleFactor, minNeighbours, flags, minSize);
	return faces;
}