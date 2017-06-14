#include "Classifiers.h"


bool PhogSVMClassifier::load(const string& genderModel, const string& ageModel){
	try{

		string delimiter = ";";

		int genderDelimiterPos = genderModel.find(delimiter);
		string g1 = genderModel.substr(0, genderDelimiterPos);
		string g2 = genderModel.substr(genderDelimiterPos + 1, genderModel.size());


		int ageDelimiterPos = ageModel.find(delimiter);
		string a1 = ageModel.substr(0, ageDelimiterPos);
		string a2 = ageModel.substr(ageDelimiterPos + 1, ageModel.size());


		svmGender = StatModel::load<SVM>(g1);
		svmGender2 = StatModel::load<SVM>(g2);
		svmAge = StatModel::load<SVM>(a1);
		svmAge2 = StatModel::load<SVM>(a2);

		return true;
	}
	catch (Exception e){
		return false;
	}
}

void PhogSVMClassifier::trainModel(const string& genderSrc, const string& ageSrc){
	string delimiter = ";";

	int genderDelimiterPos = genderSrc.find(delimiter);
	string g1 = genderSrc.substr(0, genderDelimiterPos);
	string g2 = genderSrc.substr(genderDelimiterPos + 1, genderSrc.size());


	int ageDelimiterPos = ageSrc.find(delimiter);
	string a1 = ageSrc.substr(0, ageDelimiterPos);
	string a2 = ageSrc.substr(ageDelimiterPos + 1, ageSrc.size());

	trainModelSVM(svmGender, g1, 1.0, 2.0);//cambiar
	trainModelSVM(svmGender2, g2, 1.0, 2.0);//cambiar
	trainModelSVM(svmAge, a1, 1.0, 2.0);//cambiar
	trainModelSVM(svmAge2, a2, 1.0, 2.0);//cambiar
}


void PhogSVMClassifier::trainModelSVM(Ptr <SVM> svmClassifier, const string& srcVectors, double c, float gamma){
	Ptr<TrainData> tData = TrainData::loadFromCSV(srcVectors, 0);

	svmClassifier = SVM::create();

	svmClassifier->setKernel(SVM::RBF);

	svmClassifier->setType(SVM::C_SVC);
	svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 800, 1e-6));

	svmClassifier->setC(c);
	svmClassifier->setGamma(gamma);

	svmClassifier->train(tData);
}

vector<string> PhogSVMClassifier::classiffy(cv::Mat face){

	cvtColor(face, face, CV_BGR2GRAY);
	vector<string> result;


	if (!svmGender.empty() && !svmGender2.empty() && !svmAge.empty() && !svmAge2.empty()){
		Mat features;
		PhogFeatures::computePHOG(face, 3, 16, features);

		bool isUnder250 = face.cols <= 250 && face.rows <= 250;

		float g = isUnder250 ? svmGender->predict(features.clone()) : svmGender2->predict(features.clone());
		result.push_back(to_string((int)g));


		float a = isUnder250 ? svmAge->predict(features.clone()) : svmAge2->predict(features.clone());
		string age = a == 0 ? "16-30" : (a == 1 ? "30-45" : "45+");
		result.push_back(age);
	}

	return result;
}

bool PhogSVMClassifier::usingMajority(){
	return majority;
}


bool PhogSVMClassifier::usingWeight(){
	return weighted;
}
