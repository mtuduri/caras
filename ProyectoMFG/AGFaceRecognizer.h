#pragma once

#include "IFaceDetector.h"
#include "FaceDetectors.h"
#include "IClassifier.h"
#include "Classifiers.h"
#include "ClassifierThread.h"
#include "ITracker.h"
#include "Trackers.h"

// *** NameSpaces *** 
using namespace std;

class AGRecognizer {
public:
	static void runAGRecognizer(cv::VideoCapture input, ITracker* tracker, IFaceDetector* faceDetector, IClassifier* classifier);
};
