//  ****** BASE TRACKER ******
#pragma once

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>
#include <map>

#include "Person.h"

using namespace cv;
using namespace std;

class ITracker {

public:
	map<int, Person>* trackedPersons;
	vector<int>* lostPersonsIds;

	virtual ~ITracker(){}

	virtual void initTracker(Mat frame, Mat frameGray, vector<Rect> detectedFaces) = 0;
	virtual void addDetections(Mat frame, Mat frameGray, vector<Rect> detectedFaces) = 0;
	virtual void track(Mat frame, Mat prevGray, Mat frameGray)=0;
	
};