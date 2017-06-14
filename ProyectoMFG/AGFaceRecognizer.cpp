#include "AGFaceRecognizer.h"

void verification(ITracker* tracker);

// *** CONSTANTS ***
const int FRAME_DETECTION_RATE = 10;
const int FRAME_CLASIFICATION_RATE = 10;


void AGRecognizer::runAGRecognizer(cv::VideoCapture input, ITracker* tracker, IFaceDetector* faceDetector, IClassifier* classifier){
	if (!input.isOpened()) cout << "[runAGRecognizer] ERROR: Cannot open video!\n";
	cv::Mat frame, image, prevGray, gray;
	int iter = 0;
	vector<cv::Rect> detectedFaces;

	ClassifierThread* classifierThread = new ClassifierThread(classifier, tracker->trackedPersons);
	classifierThread->start();

	while (input.read(frame)){
		frame.copyTo(image);
		cvtColor(image, gray, COLOR_BGR2GRAY);

		if (iter%FRAME_DETECTION_RATE == 0)
		{
			detectedFaces = faceDetector->detectFaces(frame, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(100, 100));
			if (iter == 0){
				tracker->initTracker(image, gray, detectedFaces);
				classifierThread->runClassification();
			}
			else{
				tracker->addDetections(image, gray, detectedFaces);
				classifierThread->runClassification();
			}

		}
		else if (iter%FRAME_CLASIFICATION_RATE == 1){
			classifierThread->runClassification();
		}
		tracker->track(image, prevGray, gray);


		cv::imshow("MFG", image);

		if (cv::waitKey(30) >= 0) {
			// only for verification REMOVE!!!!!!!!!
			verification(tracker);
			break;
		}
		cv::swap(prevGray, gray);
		iter++;
	}
	//at this point the main thread has finished
	classifierThread->runClassification();// run one last classification before to kill the thread
	classifierThread->~ClassifierThread();

	// only for verification REMOVE!!!!!!!!!
	verification(tracker);
}

int maxThree(int a, int b, int c);
int maxThree(int a, int b, int c){
	int m = a;
	(m < b) && (m = b); //these are not conditional statements.
	(m < c) && (m = c); //these are just boolean expressions.
	return m;
}
void verification(ITracker* tracker){

	cout << "\n****************** STATISTICS ****************** \n\n";
	//counters
	int male = 0;
	int female = 0;
	int young = 0;
	int adult = 0;
	int old = 0;


	map<int, Person> trackedPersons = (*tracker->trackedPersons);
	cout << "[PERSONS]: " << trackedPersons.size() << "\n\n";
	map<int, Person>::iterator person;
	for (person = trackedPersons.begin(); person != trackedPersons.end(); person++) {

		if (person->second.genderPrediction == 1)
			male++;
		else
			female++;

		if (person->second.agePrediction == "16-30")
			young++;

		else if (person->second.agePrediction == "30-45")
			adult++;
		else if (person->second.agePrediction == "45+")
			old++;

		//FACE VERIFICATION
		int i = 0;
		int to = person->second.faces.size();
		for (i; i < to; i++){

			int g = person->second.gender[i];
			string a = person->second.age[i];


			std::stringstream ss;
			ss << "C:/ProyectoCARAS/CARAS - Opencv3.0/ProyectoMFG/resources/trackedPersons/person" << person->first << "-p" << i << "-G(" << g << ")-A(" << a << ").png";

			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			try {
				imwrite(ss.str(), person->second.faces[i], compression_params);
			}
			catch (runtime_error& ex) {
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
			}
		}
	}
	cout << "\n_______GENDER_______\n\n";
	cout << "[MALE]: " << male << "\n";
	cout << "[FEMALE]: " << female << "\n";
	String gMajority = male == female ? "EQUAL" : (female > male ? "FEMALE" : "MALE");
	cout << "[GENDER MAJORITY]: " << gMajority << "\n";


	cout << "\n_______AGE_______\n\n";
	cout << "[YOUNG]: " << young << "\n";
	cout << "[ADULT]: " << adult << "\n";
	cout << "[OLD]: " << old << "\n";
	int maxAgeCount = maxThree(young, adult, old);
	if (maxAgeCount == young && young == adult && adult == old){
		cout << "[AGE MAJORITY]: EQUAL \n";
	}
	else if (maxAgeCount == young && young == adult){
		cout << "[AGE MAJORITY]: YOUNG & ADULT \n";
	}
	else if (maxAgeCount == young  && young == old){
		cout << "[AGE MAJORITY]: YOUNG & OLD \n";
	}
	else if (maxAgeCount == adult && adult == old){
		cout << "[AGE MAJORITY]: ADULT & OLD \n";
	}
	else{

		if (maxAgeCount == young){
			cout << "[AGE MAJORITY]: YOUNG \n";
		}
		else if (maxAgeCount == adult){
			cout << "[AGE MAJORITY]: ADULT \n";
		}
		else if (maxAgeCount == old){
			cout << "[AGE MAJORITY]: OLD \n";
		}

	}


	cout << "\n************** STATISTICS FINISHED **************\n\n";
	system("pause");
}