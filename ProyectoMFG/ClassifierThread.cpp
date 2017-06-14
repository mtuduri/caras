#include "ClassifierThread.h"


#include "IFaceDetector.h"
#include "FaceDetectors.h"

#define PI 3.14159265

void ClassifierThread::start(){
	the_thread = std::thread(&ClassifierThread::run, this);
}
void ClassifierThread::runClassification(){
	classify = true;
}

float maxThree(float a, float b, float c);
float maxThree(float a, float b, float c){
	float m = a;
	(m < b) && (m = b); //these are not conditional statements.
	(m < c) && (m = c); //these are just boolean expressions.
	return m;
}

void ClassifierThread::run(){

	IFaceDetector* eyesDetector = new OpenCLFaceDetector("resources\\cascades\\haarcascades\\haarcascade_eye.xml");
	IFaceDetector* eyesDetector2 = new OpenCLFaceDetector("resources\\cascades\\haarcascades\\haarcascade_mcs_eyepair_small.xml");
	IFaceDetector* noseDetector = new OpenCLFaceDetector("resources\\cascades\\haarcascades\\haarcascade_mcs_nose.xml");
	IFaceDetector* mouthDetector = new OpenCLFaceDetector("resources\\cascades\\haarcascades\\haarcascade_mcs_mouth.xml");



	while (!stop_thread){
		//run classificaion
		if (classify) {
			cout << "Runing clasification thread" << endl;
			bool isUsingMajority = classifier->usingMajority();
			bool isUsingWeight = classifier->usingWeight();
			map<int, Person>::iterator person;
			for (person = (*trackedPersons).begin(); person != (*trackedPersons).end(); person++) {
				int i = person->second.lastFaceIndex;
				int to = person->second.faces.size();
				if (i > to)
					break;


				float maleWeight = 0.0;
				float femaleWeight = 0.0;

				float youngWeight = 0.0;
				float adultWeight = 0.0;
				float oldWeight = 0.0;

				for (i; i < to; i++){
					Mat face = person->second.faces[i];
					float minWeight = 0.0;

					if (isUsingWeight){


						//FACE POSITION DETECTION
						Mat test = face.clone();
						vector<Rect> eyes;
						eyes = eyesDetector->detectFaces(test, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
						if (eyes.size() < 2)
							eyes = eyesDetector2->detectFaces(test, 1.05, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
						Point l_eye_c, r_eye_c, nose_c;
						//EYES DETECTION
						if (eyes.size() > 1){


							Rect _leftEye = eyes[0].x < eyes[1].x ? eyes[0] : eyes[1];
							Rect _rightEye = eyes[0].x < eyes[1].x ? eyes[1] : eyes[0];

							l_eye_c = Point(_leftEye.x + _leftEye.width / 2, _leftEye.y + _leftEye.height / 2);
							rectangle(test, _leftEye, Scalar(255, 0, 0), 2);

							r_eye_c = Point(_rightEye.x + _rightEye.width / 2, _rightEye.y + _rightEye.height / 2);
							rectangle(test, _rightEye, Scalar(255, 0, 0), 2);


							//NOSE DETECTION
							vector<Rect> nose;
							nose = noseDetector->detectFaces(test, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
							if (nose.size() < 1){
								nose = mouthDetector->detectFaces(test, 1.05, 6, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(10, 10));
							}
							if (nose.size() > 0)
							{
								Rect _nose = nose[0];
								nose_c = Point(_nose.x + _nose.width / 2, _nose.y + _nose.height / 2);
								rectangle(test, _nose, Scalar(255, 0, 0), 2);

								circle(test, nose_c, 1, Scalar(0, 0, 255), 3);


								//angle calculation  
								float eyesAngle = l_eye_c.x - r_eye_c.x != 0 ? atan((l_eye_c.y - r_eye_c.y) / (l_eye_c.x - r_eye_c.x)) : PI / 2;
								Point eyes_c = Point((l_eye_c.x + r_eye_c.x) / 2, (l_eye_c.y + r_eye_c.y) / 2);
								circle(test, eyes_c, 1, Scalar(0, 0, 255), 3);

								float noseAngle = eyes_c.x - nose_c.x != 0 ? atan((eyes_c.y - nose_c.y) / (eyes_c.x - nose_c.x)) : PI / 2;



								float eyesWeight = 0;

								if (eyesAngle >= -PI / 12 && eyesAngle <= PI / 12){
									//100
									eyesWeight = 100;
								}
								else if (eyesAngle >= -PI / 6 && eyesAngle< -PI / 12 || eyesAngle > PI / 12 && eyesAngle <= PI / 6){
									//50
									eyesWeight = 50;
								}


								float noseWeight = 0;
								if (noseAngle >= 5 * PI / 12 && noseAngle <= PI / 2 || noseAngle <= -5 * PI / 12 && noseAngle >= -PI / 2){
									//100
									noseWeight = 100;
								}
								else if (noseAngle > -5 * PI / 12 && noseAngle <= -3 * PI / 12 || noseAngle >= 3 * PI / 12 && noseAngle < 5 * PI / 12){
									//50
									noseWeight = 50;
								}

								minWeight = eyesWeight < noseWeight ? eyesWeight : noseWeight;


							//	imwrite("C://ProyectoCARAS//CARAS - Opencv3.0//ProyectoMFG//resources//trackedPersons//p-(" + std::to_string(person->first) + "), [" + to_string((int)minWeight) + " ] " + to_string(i) + ".jpg", test);
							}

						}
					}

					//predict age and gender
					vector<string> prediction = classifier->classiffy(face);
					if (prediction.size() == 2){


						//GENDER
						int g = stoi(prediction[0]);
						if (g == 1) (*trackedPersons)[person->first].maleMajorityCount += 1;
						int maleCount = (*trackedPersons)[person->first].maleMajorityCount;
						(*trackedPersons)[person->first].gender.push_back(g);

						if (g == 0){
							femaleWeight += minWeight;
						}
						else{
							maleWeight += minWeight;
						}

						//AGE
						string ageRange = prediction[1];
						(*trackedPersons)[person->first].age.push_back(ageRange);
						if (ageRange == "16-30"){
							youngWeight += minWeight;
						}
						else if (ageRange == "30-45"){
							adultWeight += minWeight;
						}
						else{
							oldWeight += minWeight;
						}


						if (!isUsingMajority && !isUsingWeight){
							(*trackedPersons)[person->first].genderPrediction = g;
							(*trackedPersons)[person->first].agePrediction = ageRange;
						}


						//imwrite("C://ProyectoCARAS//CARAS - Opencv3.0//ProyectoMFG//resources//trackedPersons//PERSON-[" + std::to_string(person->first) + "]-" + std::to_string(i) + "(" + std::to_string(g) + ").jpg", face);
					}

				}

				//WEIGHT CLASSIFICATION

				if (isUsingWeight){

					(*trackedPersons)[person->first].femaleWeight += femaleWeight;
					(*trackedPersons)[person->first].maleWeight += maleWeight;

					if (((*trackedPersons)[person->first].femaleWeight) / to != ((*trackedPersons)[person->first].maleWeight) / to){

						int weightedGender = ((*trackedPersons)[person->first].femaleWeight) / to > ((*trackedPersons)[person->first].maleWeight) / to ? 0 : 1;
						(*trackedPersons)[person->first].genderPrediction = weightedGender;
					}
					else{
						//GENDER MAJORITY
						(*trackedPersons)[person->first].genderPrediction = to - (*trackedPersons)[person->first].maleMajorityCount > (*trackedPersons)[person->first].maleMajorityCount ? 0 : 1;

					}


					(*trackedPersons)[person->first].youngWeight += youngWeight;
					(*trackedPersons)[person->first].adultWeight += adultWeight;
					(*trackedPersons)[person->first].oldWeight += oldWeight;

					string weightedAge;
					float young = ((*trackedPersons)[person->first].youngWeight) / to;
					float adult = ((*trackedPersons)[person->first].adultWeight) / to;
					float old = ((*trackedPersons)[person->first].oldWeight) / to;
					float maxAgeW = maxThree(young, adult, old);

					if (maxAgeW == young == adult == old || maxAgeW == young == adult || maxAgeW == adult == old || maxAgeW == young == old){
						//AGE MAJORITY
						string popular = "n";
						vector<string> age = (*trackedPersons)[person->first].age;
						if (age.size() > 1){
							int count = 0;
							for (int x = 0; x < age.size(); x++){
								string range = age[x];
								int rangeCount = 0;
								for (int n = 1; n < age.size(); n++){
									if (range == age[n])
										rangeCount++;
								}
								if (rangeCount > count){
									popular = range;
									count = rangeCount;
								}

							}
						}
						(*trackedPersons)[person->first].agePrediction = popular == "n" &&  age.size()>0 ? age[age.size() - 1] : popular;
					}
					else{

						if (maxAgeW == young){
							weightedAge = "16-30";
						}
						else if (maxAgeW == adult){
							weightedAge = "30-45";
						}
						else{
							weightedAge = "45+";
						}
						(*trackedPersons)[person->first].agePrediction = weightedAge;
					}
				}

				else if (isUsingMajority){

					//GENDER MAJORITY
					(*trackedPersons)[person->first].genderPrediction = to - (*trackedPersons)[person->first].maleMajorityCount > (*trackedPersons)[person->first].maleMajorityCount ? 0 : 1;


					//AGE MAJORITY
					string popular = "n";
					vector<string> age = (*trackedPersons)[person->first].age;
					if (age.size() > 1){
						int count = 0;
						for (int x = 0; x < age.size(); x++){
							string range = age[x];
							int rangeCount = 0;
							for (int n = 1; n < age.size(); n++){
								if (range == age[n])
									rangeCount++;
							}
							if (rangeCount > count){
								popular = range;
								count = rangeCount;
							}

						}
					}
					(*trackedPersons)[person->first].agePrediction = popular == "n" &&  age.size()>0 ? age[age.size() - 1] : popular;

				}

				//UPD last face index
				person->second.lastFaceIndex = to;
			}
			classify = false;
		}
	}
}
