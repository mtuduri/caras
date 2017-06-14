#pragma once
#include <iostream>
#include <vector>
#include "core.hpp"

using namespace std;
using namespace cv;


class Person {
public:
	Person() {}
	Person(Rect faceRect, vector<Mat> faces, int score, vector<Point2f> facePoints, int active) : faceRect(faceRect), faces(faces), score(score), facePoints(facePoints), active(active) {}
	Rect faceRect; // only for compare correlation with prev frame
	vector<Mat> faces; // save faces to classify 
	int lastFaceIndex = 0;

	//Weighted classification
	float maleWeight = 0.0;
	float femaleWeight = 0.0;

	float youngWeight = 0.0;
	float adultWeight = 0.0;
	float oldWeight = 0.0;

	vector<int> gender; // gender classification for each face
	int maleMajorityCount = 0; // male label counter
	int genderPrediction = 0;

	vector<string> age; // age classification for each face
	string agePrediction = "";

	int score;
	int active; //flag to improve correlation comparition, so we can discard non active persons 
	vector<Point2f> facePoints;
};