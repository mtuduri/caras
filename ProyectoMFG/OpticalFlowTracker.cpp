#include "Trackers.h"
#include <limits>

cv::Scalar FEMALE_COLOR = cv::Scalar(255, 0, 255);
cv::Scalar MALE_COLOR = cv::Scalar(255, 0, 0);
cv::Scalar ID_COLOR = cv::Scalar(0, 255, 0);
cv::Scalar AGE_COLOR = cv::Scalar(0, 220, 0);

//***************** Vars definition ***************** 
int highScore = 1, lastId = 0;
vector<Point2f> trackedFacesPoints[2];

//Optical flow termination citeria
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
Size subPixWinSize(10, 10), winSize(31, 31);
const int MAX_COUNT = 500;


//***************** Main methods *****************

void  OpticalFlowTracker::initTracker(Mat frame, Mat frameGray, vector<Rect> detectedFaces){
	for (int f = 0; f < detectedFaces.size(); f++){
		// first initialization  add all!!
		cv::Mat face = cv::Mat(frame, detectedFaces[f]);
		addPerson(getNextId(), face, detectedFaces[f], 1, frameGray, true);
	}
}
Person mergePerson(Person p1, Person p2);
void OpticalFlowTracker::addDetections(Mat frame, Mat frameGray, vector<Rect> detectedFaces){
	trackedFacesPoints[0].clear();
	trackedFacesPoints[1].clear();
	highScore++;
	for (int f = 0; f < detectedFaces.size(); f++){


		Mat face = Mat(frame, detectedFaces[f]);
		int faceId = findMatchingFace(face, detectedFaces[f]);

		if (faceId != -1){
			//Face already tracked
			updatePerson(faceId, face, detectedFaces[f], frameGray, 1);
		}
		else{

			double maxCorrelation = 0.0;
			int personId = -1;
			//find if the same object were tracked and merge it 
			map<int, Person>::iterator person2;
			for (person2 = (*trackedPersons).begin(); person2 != (*trackedPersons).end(); person2++){

				if (person2->second.active == 0){ // si esta inactiva
					for (int i = 0; i < person2->second.faces.size(); i++){
						double correlation = compareHistorgrams(face, person2->second.faces[i]);
						if (correlation > 0.94 && correlation > maxCorrelation){
							maxCorrelation = correlation;
							personId = person2->first;
						}
					}

				}

			}
			if (personId != -1){
				updatePerson(personId, face, detectedFaces[f], frameGray, 1);
				(*trackedPersons)[personId].score = highScore;

			}
			else{
				//New Face detected
				addPerson(getNextId(), face, detectedFaces[f], highScore, frameGray, false);

			}



		}
	}


	vector<int>personIdsToRemove;
	map<int, Person>::iterator person;
	for (person = (*trackedPersons).begin(); person != (*trackedPersons).end(); person++){

		if (person->second.active == 1){
			// Determine which tracked objects were lost
			int diff = highScore - person->second.score;
			if (diff > 2) {
				(*trackedPersons)[person->first].active = 0;
				(*lostPersonsIds).push_back(person->first);
			}
			else{
				trackedFacesPoints[1].insert(trackedFacesPoints[1].end(), (*trackedPersons)[person->first].facePoints.begin(), (*trackedPersons)[person->first].facePoints.end());
			}


		}

	}

}



void OpticalFlowTracker::track(Mat frame, Mat prevGray, Mat frameGray){
	if (!trackedFacesPoints[0].empty())
	{
		vector<uchar> status;
		vector<float> err;

		if (prevGray.empty())
			frameGray.copyTo(prevGray);

		calcOpticalFlowPyrLK(prevGray, frameGray, trackedFacesPoints[0], trackedFacesPoints[1], status, err, winSize,
			3, termcrit, 0, 0.001);

		int minX = numeric_limits<int>::max(), minY = numeric_limits<int>::max(), maxX = 0, maxY = 0;

		int from = 0, to = 0;
		vector<Point2f> newPoints;

		map<int, Person>::iterator person;
		for (person = (*trackedPersons).begin(); person != (*trackedPersons).end(); person++) {
			int diff = highScore - person->second.score;

			if (person->second.active == 1){

				Scalar genderColor = person->second.genderPrediction == 1 ? MALE_COLOR : FEMALE_COLOR;

				if (person->second.facePoints.size() > 0){

					int pointsSize = person->second.facePoints.size();
					to = from + pointsSize;

					vector<Point2f>inliers;
					ransac(person->second.facePoints, person->second.faceRect, vector<Point2f>(trackedFacesPoints[1].begin() + from, trackedFacesPoints[1].begin() + to), inliers, 200);

					for (int i = 0; i < inliers.size(); i++)
					{
						//if (status[i] == 1){
						//	if (diff < 1) {
						if (inliers[i].x > maxX) maxX = inliers[i].x;
						if (inliers[i].y > maxY) maxY = inliers[i].y;
						if (inliers[i].x < minX) minX = inliers[i].x;
						if (inliers[i].y < minY) minY = inliers[i].y;
						//	}

						newPoints.push_back(inliers[i]);
						if (diff < 1) {
							//draw tracked points
							//circle(frame, inliers[i], 5, genderColor, -1, 8);
						}
						//	}
					}

					(*trackedPersons)[person->first].facePoints = inliers;
					Rect boundingBox;
					if (inliers.size()>0 && diff < 1){
						boundingBox = Rect(Point(minX, minY), Point(maxX, maxY));
						(*trackedPersons)[person->first].faceRect = boundingBox;

					}
					else{
						(*trackedPersons)[person->first].active = 0;
					}

					from += pointsSize;
					if (inliers.size() > 0 && diff < 1) {
						//DRAW bounding box and labels
						rectangle(frame, boundingBox, genderColor, 4);
						putText(frame, std::to_string(person->first), Point(minX, minY), FONT_HERSHEY_PLAIN, 3.5, ID_COLOR, 5.5);
						string age = person->second.agePrediction;
						if (!age.empty())
							putText(frame, age, Point(maxX, maxY), FONT_HERSHEY_PLAIN, 2.0, AGE_COLOR, 4.0);

					}
					minX = numeric_limits<int>::max(), minY = numeric_limits<int>::max(), maxX = 0, maxY = 0;

				}
				else{
					(*trackedPersons)[person->first].active = 0;
				}


			}
		}
		trackedFacesPoints[1].clear();
		trackedFacesPoints[1] = newPoints;
	}

	swap(trackedFacesPoints[1], trackedFacesPoints[0]);
}


//***************** Aux methods *****************
void OpticalFlowTracker::addPerson(int id, Mat face, Rect faceRect, int score, Mat frame, bool addPoints){
	Mat temp(frame, faceRect);
	vector<Point2f> facepoints = calculateFacePoints(temp, faceRect);
	cornerSubPix(frame, facepoints, subPixWinSize, Size(-1, -1), termcrit);
	vector<Mat> faces;
	faces.push_back(face.clone());
	(*trackedPersons).insert(map<int, Person>::value_type(id, Person(faceRect, faces, score, facepoints, 1)));

	if (addPoints)
		trackedFacesPoints[1].insert(trackedFacesPoints[1].end(), facepoints.begin(), facepoints.end());
}

void OpticalFlowTracker::updatePerson(int faceId, Mat face, Rect faceRet, Mat frameGray, int active){
	(*trackedPersons)[faceId].score += 1;
	Mat temp(frameGray, faceRet);
	vector<Point2f> facepoints = calculateFacePoints(temp, faceRet); // calculate new points 
	cornerSubPix(frameGray, facepoints, subPixWinSize, Size(-1, -1), termcrit);
	(*trackedPersons)[faceId].faceRect = faceRet; //update Person's faceRect
	(*trackedPersons)[faceId].faces.push_back(face.clone()); //add face detected
	//replace new points
	(*trackedPersons)[faceId].facePoints = facepoints;
	(*trackedPersons)[faceId].active = active;
}

int OpticalFlowTracker::findMatchingFace(Mat actualface, Rect actualRect){
	double maxCorrelation = 0.0;
	int personId = -1;

	//pto medio de la cara persona a buscar
	Point center(actualRect.x + actualRect.width / 2, actualRect.y + actualRect.height / 2);
	float eps = (actualRect.width + actualRect.height) / 2;

	map<int, Person>::iterator person;
	for (person = (*trackedPersons).begin(); person != (*trackedPersons).end(); person++) {

		if (person->second.active == 1){ // compare only with active persons

			Rect pfaceRect = person->second.faceRect; //pto medio de la cara persona 2
			Point pCenter(pfaceRect.x + pfaceRect.width / 2, pfaceRect.y + pfaceRect.height / 2);

			if (abs(center.x - pCenter.x) < eps && abs(center.y - pCenter.y) < eps){
				for (int i = 0; i < person->second.faces.size(); i++){
					double actualCorrelation = compareHistorgrams(actualface, person->second.faces[i]);
					if (actualCorrelation > 0.4 && actualCorrelation > maxCorrelation){
						maxCorrelation = actualCorrelation;
						personId = person->first;
					}
				}

			}
		}

	}
	return personId;
}

double OpticalFlowTracker::compareHistorgrams(Mat f1, Mat f2){

	Mat  hsvF1;
	Mat  hsvF2;

	cv::cvtColor(f1, hsvF1, CV_BGR2HSV);
	cv::cvtColor(f2, hsvF2, CV_BGR2HSV);

	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	int channels[] = { 0, 1 };

	MatND hist_base;
	MatND hist_test;

	calcHist(&hsvF1, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsvF2, 1, channels, Mat(), hist_test, 2, histSize, ranges, true, false);
	normalize(hist_test, hist_test, 0, 1, NORM_MINMAX, -1, Mat());


	// comparar similitud  con correlation si es 1 es 100%
	try{
		return compareHist(hist_base, hist_test, 0); // Methods 0- Correlation 1-chi square 2-intersection 3-bhattacharyya
	}
	catch (Exception e){

	}
	return 0;

}

vector<Point2f> OpticalFlowTracker::calculateFacePoints(Mat face, Rect faceBox){
	vector<Point2f> facepoints;
	// geet good points to track from face detected.
	goodFeaturesToTrack(face, facepoints, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
	for (int i = 0; i < facepoints.size(); i++){
		facepoints[i].x += faceBox.x;
		facepoints[i].y += faceBox.y;
	}
	return facepoints;
}

int OpticalFlowTracker::getNextId(){
	lastId++;
	return lastId;
}

void OpticalFlowTracker::ransac(vector<Point2f>oldPoints, Rect oldRect, vector<Point2f>newPoints, vector<Point2f> &inliers, int maxIter){

	if (newPoints.size() > 3){
		float eps = 0.12*oldRect.height;
		float threshold = 0.95;

		for (int n = 0; n < maxIter; n++){
			int randomIndex = (rand() % (int)(newPoints.size()));
			if (randomIndex > newPoints.size() || randomIndex > oldPoints.size()){
				cout << "";
			}
			Point2f randomNewPoint1 = newPoints[randomIndex];
			Point2f randomOldPoint1 = oldPoints[randomIndex];

			randomIndex = (rand() % (int)(newPoints.size()));
			Point2f randomNewPoint2 = newPoints[randomIndex];
			Point2f randomOldPoint2 = oldPoints[randomIndex];

			randomIndex = (rand() % (int)(newPoints.size()));
			Point2f randomNewPoint3 = newPoints[randomIndex];
			Point2f randomOldPoint3 = oldPoints[randomIndex];

			Point2f randomNewPoint;
			randomNewPoint.x = (randomNewPoint1.x + randomNewPoint2.x + randomNewPoint3.x) / 3;
			randomNewPoint.y = (randomNewPoint1.y + randomNewPoint2.y + randomNewPoint3.y) / 3;
			Point2f randomOldPoint;
			randomOldPoint.x = (randomOldPoint1.x + randomOldPoint2.x + randomOldPoint3.x) / 3;
			randomOldPoint.y = (randomOldPoint1.y + randomOldPoint2.y + randomOldPoint3.y) / 3;

			Rect newRect = oldRect;
			newRect.x += (randomNewPoint.x - randomOldPoint.x) - eps;
			newRect.y += (randomNewPoint.y - randomOldPoint.y) - eps;
			newRect.width += eps * 2;
			newRect.height += eps * 2;

			float borderRight = newRect.x + newRect.width;
			float borderdown = newRect.y + newRect.height;

			inliers.clear();
			for (int i = 0; i < newPoints.size(); i++){
				Point2f p = newPoints[i];
				if (p.x >= newRect.x && p.x <= borderRight && p.y >= newRect.y && p.y <= borderdown){
					//inlier
					inliers.push_back(p);
				}
			}
			if ((inliers.size() / (float)newPoints.size()) > threshold){
				break;
			}
		}


	}


}