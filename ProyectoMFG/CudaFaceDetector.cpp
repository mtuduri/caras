#include "FaceDetectors.h"

void CudaFaceDetector::init(cv::String harscasade){
	cascade_gpu = cv::cuda::CascadeClassifier::create(harscasade);
}

vector<cv::Rect> CudaFaceDetector::detectFaces(cv::Mat frame, double scaleFactor = 1.1000000000000001, int minNeighbours = 3, int flags = 0, cv::Size minSize = cv::Size(30, 30)){
	cv::cuda::GpuMat image_gpu, image_gpu_gray, objbuf;
	vector<cv::Rect>faces;
	image_gpu.upload(frame);
	cv::cuda::cvtColor(image_gpu, image_gpu_gray, CV_BGR2GRAY); // Convert to gray
	cv::cuda::equalizeHist(image_gpu_gray, image_gpu_gray);          // Equalize histogram
	cascade_gpu->detectMultiScale(image_gpu_gray, objbuf);
	cascade_gpu->convert(objbuf, faces);
	return faces;
}
