#include "Classifiers.h"
#include "ClassifierUtils.h"

//Constants
string MODEL_PATH = "resources//model//modelo.yml";
int MALE_LABEL = 49;

string TRAINING_CSV_PATH = "resources\\datasets\\MORPH_nonCommercial\\salidaShuffled.csv"; // path csv con imgs y metadata
string LABALES_PATH = "resources\\datasets\\Morph\\MORPH_nonCommercial\\labels.txt";//path donde guardo el txt con los labels
string IMAGES_PATH = "resources\\datasets\\Morph\\MORPH_nonCommercial\\Album2 procesado";//path donde guardo las1 imagenes procesadas
int DATA_SET_SIZE = 1000;

// These vectors hold the images and corresponding labels.
vector<cv::Mat> images;
vector<int> labels;


bool BasicGenderClassifier::load(const string& genderModel, const string& ageModel){
	model->load(genderModel);
	return true;
}

void BasicGenderClassifier::trainModel(const string& genderSrc, const string& ageSrc){
	trainBasicModel(IMAGES_PATH, LABALES_PATH, DATA_SET_SIZE);
}

void BasicGenderClassifier::trainBasicModel(const string& srcImgs, const string& srcLabels, int size){

	//leer imagenes y labels desde src
	ClassiFierUtils::loadImgs(srcImgs, images, size);
	ClassiFierUtils::loadLabels(srcLabels, labels, size);


	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}

	int height = images[0].rows;

	//Pasamos aprox 1/3 de las muestras para test.
	cout << "Particionando el dataset" << endl;
	int testSize = (int)images.size() / 3;
	vector<cv::Mat> testImages;
	vector<int> testLabels;
	cv::Mat testSample;
	int testLabel;
	for (int i = 0; i < testSize; i++)
	{
		testSample = images[images.size() - 1];
		testLabel = labels[labels.size() - 1];
		testImages.push_back(testSample);
		testLabels.push_back(testLabel);
		images.pop_back();
		labels.pop_back();
	}

	cout << "Entrenando el modelo" << endl;

	model->train(images, labels);
	model->save(MODEL_PATH);

	// The following line predicts the label of a given
	// test image:
	int count = 0;
	cout << "Comenzando clasificación del conjunto de test" << endl;

	for (int i = 0; i < testSize; i++)
	{
		testLabel = testLabels[testLabels.size() - 1];
		testSample = testImages[testImages.size() - 1];
		testLabels.pop_back();
		testImages.pop_back();
		int predictedLabel = model->predict(testSample);

		string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
		cout << result_message << endl;
		if (predictedLabel == testLabel) { count++; }
	}

	cout << "Tasa de acierto: " << (float)count / testSize << "\n";
	std::ofstream outfile;
	outfile.open("Tasa de acierto.txt", std::ios_base::trunc);
	outfile << (float)count / testSize;
}


vector<string> BasicGenderClassifier::classiffy(cv::Mat face){
	vector<string> result;

	resize(face, face, cv::Size(100, 120), 0, 0, cv::INTER_CUBIC);;
	cvtColor(face, face, COLOR_BGR2GRAY);
	cv::equalizeHist(face, face);// Equalize histogram
	string g = model->predict(face) == MALE_LABEL ? "1" : "0";
	result.push_back(g);

	result.push_back("20-30");

	return result;
}

bool BasicGenderClassifier::usingMajority(){
	return majority;
}

bool BasicGenderClassifier::usingWeight(){
	return weighted;
}