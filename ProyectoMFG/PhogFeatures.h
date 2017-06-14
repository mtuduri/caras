#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

// *** NameSpaces *** 
using namespace cv;

class PhogFeatures {
public:
	static void computePHOG(cv::Mat inputImg, int levels, int binNumber, cv::Mat &phog);
private:
	static void buildSubLevelHistograms(cv::Mat superLevelHistograms, int previousLevel, int histogramSize, cv::Mat &subLevelHistogramsOut);
};