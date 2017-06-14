#include "AGFaceRecognizer.h"

// *** CONSTANTS *** 
bool CAPTURE_FROM_WEBCAM = false;
string SOURCE_VIDEO = "resources\\video\\vir2.mp4";

//  *** face detection constants *** 
string LBP_CASCADE = "resources\\cascades\\lbpcascades\\lbpcascade_frontalface.xml";
bool LBP_CASCADE_ENABLED = false;
string HAAR_CASCADE = "resources\\cascades\\haarcascades\\haarcascade_frontalface_alt.xml";
string CUDA_HAAR_CASCADE = "resources\\cascades\\haarcascades_cuda\\haarcascade_frontalface_alt.xml";
bool CUDA_ENABLED = false;

// *** Classifier constants *** 
bool PHOG_ENABLED = true;
string BASIC_GENDER_MODEL_PATH = "resources//model//modelo.yml";
string BASIC_AGE_MODEL_PATH = "";
string PHOG_GENDER_MODEL_PATH = "resources\\model\\gender_svm_11C_for_vir2(probar).xml;resources\\model\\gender_svm_MF10k_High_C.xml";
string PHOG_AGE_MODEL_PATH = "resources\\model\\age_svm_12B_250.xml;resources\\model\\age_svm_MF10k_6_full.xml";
bool MAJORITY_CLASIFICATION = true;
bool WEIGHTED_CLASSIFICATION = true;


int main()
{
	cout << "****************** INITIALIZATION ****************** \n\n";

	cout << "[TRACKER]: Optical Flow tracker \n\n";
	ITracker * tracker = new OpticalFlowTracker();

	IClassifier* classifier;
	cout << "[CLASSFIER]: loading classifier... \n";

	if (PHOG_ENABLED){
		cout << "[PHOG CLASSFIER]: Loading age and gender model . . . \n\n";
		classifier = new PhogSVMClassifier(MAJORITY_CLASIFICATION, WEIGHTED_CLASSIFICATION);
		classifier->load(PHOG_GENDER_MODEL_PATH, PHOG_AGE_MODEL_PATH);
	}
	else{
		cout << "[BASIC CLASSFIER]: Loading age and gender model . . . \n\n";
		cv::Ptr<cv::face::FaceRecognizer> faceRecognizer = cv::face::createFisherFaceRecognizer();
		classifier = new BasicGenderClassifier(MAJORITY_CLASIFICATION, WEIGHTED_CLASSIFICATION, faceRecognizer);
		classifier->load(BASIC_GENDER_MODEL_PATH, BASIC_AGE_MODEL_PATH);
	}

	cout << "[FACE DETECTOR]: Loading cascade . . .\n";
	IFaceDetector* faceDetector;
	if (cv::cuda::getCudaEnabledDeviceCount() > 0 && CUDA_ENABLED){
		cout << "[FACE DETECTOR]: Cuda device found, setting CUDA_FACE_DETECOR \n\n";
		faceDetector = new CudaFaceDetector(CUDA_HAAR_CASCADE);
	}
	else{
		cout << "[FACE DETECTOR]: NO Cuda device found, setting OPENCL_FACE_DETECOR \n";
		if (LBP_CASCADE_ENABLED){
			faceDetector = new OpenCLFaceDetector(LBP_CASCADE);
			cout << "[FACE DETECTOR]: cascade-> LBP_CASCADE \n\n";
		}
		else{
			faceDetector = new OpenCLFaceDetector(HAAR_CASCADE);
			cout << "[FACE DETECTOR]: cascade-> HAAR_CASCADE \n\n";
		}
	}

	if (CAPTURE_FROM_WEBCAM)
		cout << "[WEBCAM]: running from webcam \n\n";
	else
		cout << "[VIDEO]: " + SOURCE_VIDEO + "\n\n";


	cout << "************** INITIALIZATION FINISHED **************\n\n";
	system("pause");
	cout << "\n\n";

	if (CAPTURE_FROM_WEBCAM){
		cv::VideoCapture cap(0);
		AGRecognizer::runAGRecognizer(cap, tracker, faceDetector, classifier);
	}
	else{
		cv::VideoCapture cap(SOURCE_VIDEO);
		cap.set(CAP_PROP_FORMAT, CV_32FC3);
		AGRecognizer::runAGRecognizer(cap, tracker, faceDetector, classifier);
	}
	return 0;
}