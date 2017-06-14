#include "PhogFeatures.h"


cv::Mat LoGKernel = (cv::Mat_<float>(7, 7) <<
	0.0048, 0.0088, 0.0120, 0.0128, 0.0120, 0.0088, 0.0048,
	0.0088, 0.0132, 0.0082, 0.0023, 0.0082, 0.0132, 0.0088,
	0.0120, 0.0082, -0.0232, -0.0471, -0.0232, 0.0082, 0.0120,
	0.0128, 0.0023, -0.0471, -0.0829, -0.0471, 0.0023, 0.0128,
	0.0120, 0.0082, -0.0232, -0.0471, -0.0232, 0.0082, 0.0120,
	0.0088, 0.0132, 0.0082, 0.0023, 0.0082, 0.0132, 0.0088,
	0.0048, 0.0088, 0.0120, 0.0128, 0.0120, 0.0088, 0.0048);


cv::Mat blur100 = (cv::Mat_<float>(7, 7) <<
	0.0008, 0.0030, 0.0065, 0.0084, 0.0065, 0.0030, 0.0008,
	0.0030, 0.0108, 0.0232, 0.0299, 0.0232, 0.0108, 0.0030,
	0.0065, 0.0232, 0.0498, 0.0643, 0.0498, 0.0232, 0.0065,
	0.0084, 0.0299, 0.0643, 0.0830, 0.0643, 0.0299, 0.0084,
	0.0065, 0.0232, 0.0498, 0.0643, 0.0498, 0.0232, 0.0065,
	0.0030, 0.0108, 0.0232, 0.0299, 0.0232, 0.0108, 0.0030,
	0.0008, 0.0030, 0.0065, 0.0084, 0.0065, 0.0030, 0.0008);

cv::Mat blur200 = (cv::Mat_<float>(7, 7) <<
	0.0113, 0.0149, 0.0176, 0.0186, 0.0176, 0.0149, 0.0113,
	0.0149, 0.0197, 0.0233, 0.0246, 0.0233, 0.0197, 0.0149,
	0.0176, 0.0233, 0.0275, 0.0290, 0.0275, 0.0233, 0.0176,
	0.0186, 0.0246, 0.0290, 0.0307, 0.0290, 0.0246, 0.0186,
	0.0176, 0.0233, 0.0275, 0.0290, 0.0275, 0.0233, 0.0176,
	0.0149, 0.0197, 0.0233, 0.0246, 0.0233, 0.0197, 0.0149,
	0.0113, 0.0149, 0.0176, 0.0186, 0.0176, 0.0149, 0.0113);

cv::Mat blur300 = (cv::Mat_<float>(7, 7) <<
	0.0148, 0.0173, 0.0190, 0.0196, 0.0190, 0.0173, 0.0148
	, 0.0173, 0.0202, 0.0222, 0.0229, 0.0222, 0.0202, 0.0173
	, 0.0190, 0.0222, 0.0243, 0.0251, 0.0243, 0.0222, 0.0190
	, 0.0196, 0.0229, 0.0251, 0.0259, 0.0251, 0.0229, 0.0196
	, 0.0190, 0.0222, 0.0243, 0.0251, 0.0243, 0.0222, 0.0190
	, 0.0173, 0.0202, 0.0222, 0.0229, 0.0222, 0.0202, 0.0173
	, 0.0148, 0.0173, 0.0190, 0.0196, 0.0190, 0.0173, 0.0148);


//Computes the histograms vector for level L-1 using level L histograms, taking them by 4 and adding them
void PhogFeatures::buildSubLevelHistograms(cv::Mat superLevelHistograms, int previousLevel, int histogramSize, cv::Mat &subLevelHistogramsOut){

	int lastFreePosition = 0;
	for (int i = 0; i < pow(2, previousLevel) - 1; i += 2)
	{
		for (int j = 0; j < pow(2, previousLevel) - 1; j += 2)
		{
			//FIXME this index can be done adding 2*histogramSize on each iteration, and adding 2^l*histogramSize on the outer for
			int index = (i * pow(2, previousLevel) + j)*histogramSize;

			//Add superLevelHistograms from position (i,j), (i+1,j), (i,j+1), (i+1,j+1)
			cv::Mat temp = superLevelHistograms(cv::Rect(index, 0, histogramSize, 1)) +
				superLevelHistograms(cv::Rect(index + histogramSize, 0, histogramSize, 1)) +
				superLevelHistograms(cv::Rect(index + pow(2, previousLevel)*histogramSize, 0, histogramSize, 1)) +
				superLevelHistograms(cv::Rect(index + pow(2, previousLevel)*histogramSize + histogramSize, 0, histogramSize, 1));

			temp.copyTo(subLevelHistogramsOut(cv::Rect(lastFreePosition, 0, histogramSize, 1)));
			lastFreePosition += histogramSize;
		}
	}
}

//Must receive a CV_32FC1 channel Mat
void PhogFeatures::computePHOG(cv::Mat inputImg, int levels, int binNumber, cv::Mat &phog){

	//if (inputImg.cols > 100 && inputImg.cols < 200 && inputImg.rows > 100 && inputImg.rows < 200)
	//	cv::filter2D(inputImg, inputImg, -1, blur100);

	//else if (inputImg.cols > 200 && inputImg.cols < 300 && inputImg.rows > 200 && inputImg.rows < 300)
	//	cv::filter2D(inputImg, inputImg, -1, blur200);

	//else if (inputImg.cols > 300 && inputImg.rows > 300)
	//	cv::filter2D(inputImg, inputImg, -1, blur300);



	inputImg.convertTo(inputImg, CV_32FC1);
	//normalize(temp, inputImg);


	cv::Mat temp;
	//if (inputImg.cols <= 250 && inputImg.rows <= 250){
		temp = cv::Mat(250, 250, CV_32FC1);
		resize(inputImg, temp, temp.size(), 0, 0, CV_INTER_CUBIC);
		inputImg = temp;
	//}
	//else if (inputImg.cols > 250 && inputImg.rows > 250 && inputImg.cols <= 450 && inputImg.rows <= 450){
	//	temp = cv::Mat(100, 100, CV_32FC1);
	//	resize(inputImg, temp, temp.size(), 0, 0, CV_INTER_CUBIC);
	//	inputImg = temp;
	//}



	//Perform bicubic interpolation on input image

	cv::Mat LoGimg;
	//Apply Laplacian of Gaussian filter
	cv::filter2D(inputImg, LoGimg, -1, LoGKernel);
	//Compute partial derivatives
	cv::Mat Dx;
	cv::Sobel(LoGimg, Dx, -1, 1, 0, 3);
	cv::Mat Dy;
	cv::Sobel(LoGimg, Dy, -1, 0, 1, 3);
	//Compute gradient
	//Magnitude
	cv::Mat Dx2;
	cv::Mat Dy2;
	pow(Dx, 2, Dx2);
	pow(Dy, 2, Dy2);
	cv::Mat gradMagnitude;
	cv::sqrt(Dx2 + Dy2, gradMagnitude);
	//Orientation
	cv::Mat orientation = cv::Mat(Dx.rows, Dx.cols, CV_32FC1);
#define PI 3.14159265

	for (int i = 0; i < Dx.rows; i++)
	{
		for (int j = 0; j < Dx.cols; j++)
		{
			if (Dx.at<float>(i, j) != 0){
				orientation.at<float>(i, j) = atan(Dy.at<float>(i, j) / Dx.at<float>(i, j)) + PI / 2; //Add PI/2 to get [0,PI] interval.
			}
			else{
				//If gradient has no X-component, its orientation is PI/2 (90 degrees)
				orientation.at<float>(i, j) = PI;  // PI/2 + PI/2 to get the [0,PI] interval.
			}
			//If both X and Y components are 0, we don't care because the gradients with module = 0 are not taken into account 
		}

	}

	//Compute Pyramidal HOG

	//Level L
	int pyramidLevelSize = inputImg.rows / pow(2, levels);  //Size of each pyramid level (size of each "cell" of the grid-like image)
	float binAmplitude = PI / binNumber; //Angle in radians of each bin to map gradient orientations
	int histogramSize = binNumber; //Size of each histogram

	int histogramsVectorSize = 0; //Size of the vector of concatenated histograms (the whole PHOG vector)
	for (int level = 0; level <= levels; level++)
	{
		histogramsVectorSize = histogramsVectorSize + (pow(2, level)) * (pow(2, level)) * histogramSize;
	}

	cv::Mat histogramsVector = cv::Mat(1, histogramsVectorSize, CV_32FC1); //Vector of concatenated histograms (each histogram of each cell at each pyramid level
	int lastFreePosition = 0;
	for (int i = 0; i < pow(2, levels); i++)
	{
		for (int j = 0; j < pow(2, levels); j++)
		{
			cv::Mat currentCellMagnitudes;
			cv::Mat currentCellOrientations;
			cv::Mat cellHistogram = cv::Mat(1, histogramSize, CV_32FC1, float(0));

			//Cut the (i,j) image cell from the grid (magnitude and orientation image)
			currentCellMagnitudes = gradMagnitude(cv::Rect(i*pyramidLevelSize, j*pyramidLevelSize, pyramidLevelSize, pyramidLevelSize));
			currentCellOrientations = orientation(cv::Rect(i*pyramidLevelSize, j*pyramidLevelSize, pyramidLevelSize, pyramidLevelSize));
			//Loop over the current image cell
			for (int m = 0; m < pyramidLevelSize; m++)
			{
				for (int n = 0; n < pyramidLevelSize; n++)
				{
					float currentMagnitude = currentCellMagnitudes.at<float>(m, n);
					float currentOrientation = currentCellOrientations.at<float>(m, n);
					if (currentMagnitude > 0){

						//Determine to which orientation bins the gradient's module will be mapped
						float binIdx = currentOrientation / binAmplitude;
						int binIdxInt = floor(binIdx);
						float binIdxDecimal = binIdx - binIdxInt;
						//Despite atan codomain is [-PI/2,PI/2] and we add PI/2 to the result to have values in [0,PI], we still
						//can have negative orientation values because numerical aproximation errors. Then we set them to 0
						if (binIdx < 0){
							binIdxInt = 0;
							binIdxDecimal = 0;
						}
						//If currentOrientation == PI (180 degrees), the index will go out of range (binIdxInt would be equal to binNumber, and the 
						//histogram indexes are [0:binNumber-1]. But since we only care about orientation, we can set current orientation to 0.
						if (binIdxInt == histogramSize){
							binIdxInt = 0;
						}
						//Linearly distribute gradient magnitude between neighbour orientation bins
						cellHistogram.at<float>(0, binIdxInt) += currentMagnitude * (1 - binIdxDecimal);
						//Use mod operation to prevent index go out of range when adding + 1
						cellHistogram.at<float>(0, (binIdxInt + 1) % histogramSize) += currentMagnitude * (binIdxDecimal);
					}
				}

			}

			cv::Mat dst = histogramsVector(cv::Rect(lastFreePosition, 0, histogramSize, 1));
			cellHistogram.copyTo(dst);
			lastFreePosition += histogramSize;
			//histogramsVector.at((i*pow(levels,2) + j)*histogramSize) = 
		}
	}
	//Use level L histograms to compute Level L-1 histograms, then compute Level L-2 using Level L-1, and so on until 0.
	for (int l = levels; l > 0; l--) // l is the previous level
	{
		//Compute the lenght of the concatenated histograms from the previously computed level
		//FIXME save pow(2,l) in a variable to avoid computing it many times
		int previousLevelSize = pow(2, l) * pow(2, l) * histogramSize;
		//Compute the lenght of the concatenated histograms vector for the upcoming level 
		int subLevelSize = pow(2, l - 1) * pow(2, l - 1) * histogramSize;
		//Take the previous level concatenated histograms from the entire histograms vector
		cv::Mat superLevelHistograms = histogramsVector(cv::Rect(lastFreePosition - previousLevelSize, 0, previousLevelSize, 1));

		buildSubLevelHistograms(superLevelHistograms, l, histogramSize, histogramsVector(cv::Rect(lastFreePosition, 0, subLevelSize, 1)));
		lastFreePosition += subLevelSize;
	}
	//normalize using L2 norm
	normalize(histogramsVector, phog);
}