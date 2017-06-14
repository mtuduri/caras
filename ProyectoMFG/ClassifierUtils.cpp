#include "ClassiFierUtils.h"

//Constants
string MIMETYPE_JPG = ".jpg";
string HAARCASCADE = "resources\\cascades\\haarcascades\\haarcascade_frontalface_alt.xml";
string LABELS_DEST = "resources\\datasets\\MORPH_nonCommercial\\labels.txt";
string MORPH_SRC = "resources\\datasets\\Morph\\MORPH_nonCommercial";
string DOUBLE_BSLASH = "\\";
string MALE_LABEL = "M";

void ClassiFierUtils::loadImgs(const string& src, vector<cv::Mat>& images, int size){

	for (int i = 1; i <= size; i++)
	{
		cv::Mat img = cv::imread(src + DOUBLE_BSLASH + to_string(i) + MIMETYPE_JPG, CV_8UC1);
		images.push_back(img);
	}
}

void ClassiFierUtils::loadLabels(const string& src, vector<int>& labels, int size){

	std::basic_ifstream<int> file(src, std::ios::binary);

	labels = std::vector<int>((std::istreambuf_iterator<int>(file)),
		std::istreambuf_iterator<int>());

}

cv::Mat ClassiFierUtils::norm_0_255(cv::InputArray _src) {
	cv::Mat src = _src.getMat();
	// Create and return normalized image:
	cv::Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

void ClassiFierUtils::read_csv(const string& filename, const string& dst, char separator, int size) {

	OpenCLFaceDetector* openClFaceDetector = new OpenCLFaceDetector(HAARCASCADE);

	//archivo de salida de los labels
	std::ofstream outfile;
	outfile.open(LABELS_DEST, std::ios_base::trunc | std::ios_base::out);


	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, race, gender, facialHair, age, ageDiff, glasses, path;


	int setSize = size; //5000 - 85 % 3000 - 85 %  2000 - 87 %  1500 - 92 %  1200 - 93%   1000- 88%//// Reshuffle y test con 1/3 1200-89%, 1500 90.5, 2000-86%

	int count = 0;
	while (getline(file, line) && count < setSize) {
		count++;
		stringstream liness(line);


		getline(liness, race, separator);
		getline(liness, gender, separator);
		getline(liness, facialHair, separator);
		getline(liness, age, separator);
		getline(liness, ageDiff, separator);
		getline(liness, glasses, separator);
		getline(liness, path);

		string pathAbs = MORPH_SRC + DOUBLE_BSLASH + path;


		if (!path.empty() && !gender.empty()) {
			cv::Mat img;
			cv::Mat src = cv::imread(pathAbs);
			vector<cv::Rect>faces;

			try{
				cv::cvtColor(src, src, CV_BGR2GRAY);
				int neighbours = 3;
				cv::Mat detectedFace;
				int iter = 0; //hay que poner un limite de iteraciones. Por ej, si con 3 vecinos da 2 positivos y con 4 da 0, nunca termina, queda cambiando entre 3 y 4 vecinos.
				do{

					faces = openClFaceDetector->detectFaces(src, 1.1, neighbours, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

					if (faces.size() == 1){

						detectedFace = cv::Mat(src, faces[0]);
						resize(detectedFace, img, cv::Size(100, 120), 0, 0, cv::INTER_CUBIC);
						//imshow("", img);
						int label;
						gender == MALE_LABEL ? label = 1 : label = 0;

						//Escribimos la imagen procesada en disco. Esto es para procesar una sola vez y luego usar eso para entrenar.
						string name = dst + to_string(count) + MIMETYPE_JPG;
						imwrite(name, img);

						//escribimos las labels en un txt.
						outfile << label;
						break;

					}
					else{
						if (faces.size() > 1){
							neighbours++;

						}
						else{
							neighbours--;

						}
					}
					iter++;
				} while (neighbours > 0 && neighbours < 6 && iter < 4);

				if (iter == 4 || neighbours == 0 || neighbours == 6){//nunca reconoció una cara sola. Hay que elegir una entre varias. Ej la mas grande

					//detecto y me quedo con el rectangulo mas grande. Si no detecto nada, uso la imagen entera.
					openClFaceDetector->detectFaces(src, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));

					if (faces.size() > 0){
						int maxArea = faces[0].area();
						cv::Rect maxRect = faces[0];
						for each (cv::Rect rect in faces)
						{
							if (rect.area() > maxArea) { maxRect = rect; maxArea = rect.area(); }
						}
						detectedFace = cv::Mat(src, maxRect);
					}
					else{
						detectedFace = src;
					}
					resize(src, img, cv::Size(100, 120), 0, 0, cv::INTER_CUBIC);
					int label;
					gender == MALE_LABEL ? label = 1 : label = 0;
					string name = dst + to_string(count) + MIMETYPE_JPG;
					imwrite(name, img);
					//escribimos las labels en un txt.
					outfile << label;
				}
			}
			catch (cv::Exception& e) {
				cerr << "Error\". Reason: " << e.msg << endl;
				exit(1);
			}

		}

		cout << "Progreso " << ((float)count / setSize) * 100 << "%       \r";
	}
	outfile.close();
}