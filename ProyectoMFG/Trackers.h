#pragma once
#include "ITracker.h"

class OpticalFlowTracker : public ITracker
{
public:
	OpticalFlowTracker(){
		trackedPersons = new map<int, Person>();
		lostPersonsIds= new vector<int>();
	}

	//Main methdos
	virtual void initTracker(Mat frame, Mat frameGray, vector<Rect> detectedFaces);
	virtual void addDetections(Mat frame, Mat frameGray, vector<Rect> detectedFaces);
	void track(Mat frame, Mat prevGray, Mat frameGray);

private:
	//Aux methods
	void addPerson(int id, Mat face, Rect faceRect, int score, Mat frame, bool addPoints);
	void updatePerson(int faceId, Mat face, Rect faceRet, Mat frameGray, int active);
	int findMatchingFace(Mat actualface, Rect actualRect);
	double compareHistorgrams(Mat f1, Mat f2);
	vector<Point2f> calculateFacePoints(Mat face, Rect faceBox);
	int getNextId();
	void ransac(vector<Point2f>oldPoints, Rect oldRect, vector<Point2f>newPoints, vector<Point2f> &inliers, int maxIter);
};