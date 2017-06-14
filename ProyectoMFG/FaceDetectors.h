
#include "IFaceDetector.h"


////*** CUDA FACE DETECTOR ****
#include "cudaobjdetect.hpp"
#include "cudaimgproc.hpp"

class CudaFaceDetector : public IFaceDetector{
public:
	CudaFaceDetector(cv::String cascadePath)
	{
		init(cascadePath);
	}
	cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu;

	virtual void init(cv::String cascadePath);

	virtual vector<cv::Rect> detectFaces(cv::Mat frame, double scaleFactor, int minNeighbours, int flags, cv::Size minSize);
};



//*** OPENCL FACE DETECTOR ****

#include "core/ocl.hpp"
#include "opencv.hpp"

using namespace std;


class OpenCLFaceDetector : public IFaceDetector{
public:
	OpenCLFaceDetector(cv::String cascadePath)
	{
		init(cascadePath);
	}

	cv::Ptr<cv::CascadeClassifier> cascade;

	virtual void init(cv::String cascadePath);

	virtual vector<cv::Rect> detectFaces(cv::Mat frame, double scaleFactor, int minNeighbours, int flags, cv::Size minSize);
};