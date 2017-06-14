#include "IClassifier.h"
#include "Person.h"
#include <iostream>
#include <map>
#include <thread>

using namespace std;
using namespace cv;


class ClassifierThread{
public:
	ClassifierThread(IClassifier* Classifier, map<int, Person>* TrackedPersons) : the_thread() {
		classifier = Classifier;
		trackedPersons = TrackedPersons;
	}
	~ClassifierThread(){
		stop_thread = true;
		if (the_thread.joinable()) the_thread.join();
	}
	void start();
	void runClassification();

private:
	std::thread the_thread;
	bool stop_thread = false;
	bool classify = false;

	IClassifier* classifier;
	map<int, Person>* trackedPersons;

	void run();
};