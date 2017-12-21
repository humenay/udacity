// P4_Advanced_Lane_Lines.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/videoio/videoio.hpp"


using namespace cv;
using namespace std;


/*Globals*/
deque<vector<double>> lSmooth;
deque<vector<double>> rSmooth;
deque<vector<double>> lSmoothReal;
deque<vector<double>> rSmoothReal;
vector<cv::Point> allPoints;
const int SMOOTHING_WIN_SIZE = 10;
const double YM_PER_PIX = 30.0/ 720;
const double XM_PER_PIX = 3.7/700;

#define WRITE_VIDEO_ENABLED 0


/*
* The purpose of this function is to load calibration images and calculate the camera's calibration constants
*/
void calibrate(Mat & intrinsic, Mat & distCoeffs, bool saveUndistorted)
{
	Mat calibImg,grayImg,undistortedImg;
	bool foundChessboard = false;
	vector<Point2f> corners;
	vector<cv::String> files;
	vector<vector<Point3f>> object_points;		//physical position of the corners in 3d space. this has to be measured by us
	vector<vector<Point2f>> image_points;		//location of corners on in the image (2d) once the program has actual physical locations and locations
	vector<Mat> rvecs, tvecs;

	vector< Point3f > obj;
	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			//0.025423 is the estimated width/height in meters of a square on a calibration image 
			obj.push_back(Point3f((float)j * 0.02423, (float)i * 0.02423, 0));
		}
	}

	glob(".\\camera_cal\\", files);
	for (size_t i = 0; i < files.size(); i++)
	{
		calibImg = imread(files[i]);;
		corners.clear();
		foundChessboard = findChessboardCorners(calibImg, cvSize(9, 6), corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		if (foundChessboard == true)
		{
			cvtColor(calibImg, grayImg, COLOR_BGR2GRAY);
			cornerSubPix(grayImg, corners, cv::Size(5, 5), cv::Size(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER, 100, 0.1));
			drawChessboardCorners(calibImg, cv::Size(9, 6), corners, foundChessboard);
			image_points.push_back(corners);
			object_points.push_back(obj);
		}

	}//closes loop

	//initialize it to 3 x 3 matrix if it is not intitialized already
	intrinsic = Mat(3, 3, CV_32FC1); 

	//modify intrinsic matrix with whatever we know. I guessed camera spect ration is 1 
	intrinsic.ptr<float>(0)[0] = 1;
	intrinsic.ptr<float>(1)[1] = 1;

	if (object_points.size() != 0)
	{
		calibrateCamera(object_points, image_points, calibImg.size(), intrinsic, distCoeffs, rvecs, tvecs);
		if (saveUndistorted == true)
		{
			for (size_t i = 0; i < files.size(); i++)
			{
				calibImg = imread(files[i]);
				undistort(calibImg, undistortedImg, intrinsic, distCoeffs);
				cv::String path = files[i].substr(0, files[i].length() - 4);
				path += "_undistorted.jpg";
				imwrite(path, undistortedImg);
			}
		}//closes if save
	}//closes if (object_points.size() != 0)
}

void initWarpImg(Mat & M, Mat & Minv)
{
	Point2f src[4];
	Point2f dst[4];

	src[0] = Point2f(490, 482); //top left
	src[1] = Point2f(810, 482); //top right 
	src[2] = Point2f(40, 720); //bottom left
	src[3] = Point2f(1250, 720); //bottom right
	
	dst[0] = Point2f(0, 0);//top left
	dst[1] = Point2f(1279, 0); //top right
	dst[2] = Point2f(40, 720); //bottom left
	dst[3] = Point2f(1250, 720); //bottom right

	//3 x 3 matrix of a perspective transform stored in M and inverse is stored in Minv
	M = getPerspectiveTransform(src, dst);
	Minv = getPerspectiveTransform(dst, src);
}

void applyFilters(Mat in, Mat & out)
{
	Mat origHSV,tmpHSV,tmpHLS,tmpLAB,gray,sobelx;

	//Apply light filter in HLS space
	cv::cvtColor(in, tmpHLS, CV_RGB2HLS);
	inRange(tmpHLS, cv::Scalar(1, 160, 180), cv::Scalar(255, 255, 255), tmpHLS);
	out = tmpHLS;

	//Apply HSV Filters to attempt to extract white and yellow lines
	cv::cvtColor(in, origHSV, CV_RGB2HSV);
	tmpHSV = origHSV.clone();
	inRange(tmpHSV, cv::Scalar(0, 50, 240), cv::Scalar(180, 255, 255), tmpHSV);//yellow
	bitwise_or(tmpHSV, out, out);

	tmpHSV = origHSV.clone();
	inRange(tmpHSV, cv::Scalar(0, 0, 240), cv::Scalar(40, 23, 255), tmpHSV);
	bitwise_or(tmpHSV, out, out);

	tmpHSV = origHSV.clone();
	inRange(tmpHSV, cv::Scalar(150, 0, 230), cv::Scalar(180, 10, 255), tmpHSV);//white
	bitwise_or(tmpHSV, out, out);
	
	cv::cvtColor(in, tmpLAB, CV_RGB2Lab);
	inRange(tmpLAB, cv::Scalar(0,118, 0), cv::Scalar(255, 148, 255), tmpLAB);
	bitwise_not(tmpLAB, tmpLAB);
	bitwise_or(tmpLAB, out, out);

	//Apply sobel detection which is similar to Canny edge detection. This mainly helps with white lines that 
	//are farthest away
	cvtColor(in, gray, CV_RGB2GRAY);
	GaussianBlur(gray, gray, cv::Size(7, 7), 3, 3);
	Sobel(gray, sobelx, CV_8U, 1, 0);
	
	//if a matrix value is above threshold of 50 then make it 255 which is akin to 1 in a binary image
	threshold(sobelx, sobelx, 50, 255, CV_THRESH_BINARY);
	
	bitwise_or(sobelx, out,out);

}

/*
* This original version of this function can be found on 
* http://www.bragitoff.com/2015/09/c-program-for-polynomial-fit-least-squares/
*/
vector<double> mPolyfit(Mat scatter, bool convertToRealWorld)
{
	int i, j, k;
	double B[3][4] = {}; //B is the Normal matrix(augmented) that will store the equations,
	double a[3] = {}; //final coefficients
	double Y[3] = {}; //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	double X[5] = {};
	int n = 2;
	Mat nonZeroCoordinates;
	findNonZero(scatter, nonZeroCoordinates);
	int N = nonZeroCoordinates.total();
	double * x = new double[N];
	double * y = new double[N];

	vector<double> retVal;

	for (i = 0; i < N; i++)
	{
		x[i] = (double)nonZeroCoordinates.at<cv::Point>(i).x;
		y[i] = (double)nonZeroCoordinates.at<cv::Point>(i).y;
		if (convertToRealWorld == true)
		{
			x[i] = x[i] * .005286;
			y[i] = y[i] * .041667;
		}
	}


	for (i = 0; i < 2 * n + 1; i++)
	{
		X[i] = 0;
		for (j = 0; j < N; j++)
		{
			//consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3) etc
			X[i] = X[i] + pow(x[j], i);
		}
	}


	//Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
	for (i = 0; i <= n; i++)
	{
		for (j = 0; j <= n; j++)
		{
			B[i][j] = X[i + j];
		}
	}


	//consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	for (i = 0; i < n + 1; i++)
	{
		Y[i] = 0;
		for (j = 0; j < N; j++)
		{
			Y[i] = Y[i] + pow(x[j], i)*y[j];
		}
	}

	//load the values of Y as the last column of B(Normal Matrix but augmented)

	for (i = 0; i <= n; i++)
	{
		B[i][n + 1] = Y[i];
	}

	//n is made n+1 because the Gaussian Elimination part below was for n equations, 
	//but here n is the degree of polynomial and for n degree we get n+1 equations
	n = n + 1;

	//From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
	for (i = 0; i < n; i++)
	{
		for (k = i + 1; k < n; k++)
		{
			if (B[i][i] < B[k][i])
			{
				for (j = 0; j <= n; j++)
				{
					double temp = B[i][j];
					B[i][j] = B[k][j];
					B[k][j] = temp;
				}
			}
		}
	}

	//loop to perform the gauss elimination
	for (i = 0; i < n - 1; i++)
	{
		for (k = i + 1; k < n; k++)
		{
			double t = B[k][i] / B[i][i];
			for (j = 0; j <= n; j++)
			{
				B[k][j] = B[k][j] - t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
			}
		}
	}

	for (i = n - 1; i >= 0; i--)                //back-substitution
	{                        //x is an array whose values correspond to the values of x,y,z..
		a[i] = B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
		for (j = 0; j < n; j++)
		{
			if (j != i)            //then subtract all the lhs values except the coefficient of the variable whose value is being calculated
			{
				a[i] = a[i] - B[i][j] * a[j];
			}
		}
		a[i] = a[i] / B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
	}
	
	
	for (i = 0; i < n; i++)
	{ 
		retVal.push_back(a[i]);
	}
	//clean up memory
	delete x;
	delete y;

	return retVal;
}

/*
* For each hot pixel in a window generate a random variable +-40 centered around the hot pixel
*/
void generateFakeData(Mat & left, Mat & right, int windowHeight, int windowStartY, int leftWindowIdx, int rightWindowIdx)
{
	int randNumL = 0;
	int randNumR = 0;
	int xLeft = 0;
	int xRight = 0;
	int y = 0;

	for (int i = 0; i < (windowHeight); i++)
	{

		randNumL = rand() % (80) - 40;
		randNumR = rand() % (80) - 40;
		xLeft = randNumL + leftWindowIdx;
		xRight = randNumR + rightWindowIdx;
		y = windowStartY - i;
		//should gauruntee a data point is generated for every y value but this is not necessarily the case  
		//as the code is currently written
		if (xLeft < 1280 && xLeft >= 0)
		{
			left.at<uchar>(cv::Point(xLeft, y)) = 255;
		}
		if (xRight < 1280 && xRight >= 0)
		{
			right.at<uchar>(cv::Point(xRight, y)) = 255;
		}
	}
}

vector<cv::Point> combinePoints(vector<double> p1, vector<double> p2)
{
	vector<cv::Point> retList;
	for (int i = 0; i < 720; i++)
	{
		double xL = p1[0] + p1[1] * i + p1[2] * i*i;
		double xR = p2[0] + p2[1] * i + p2[2] * i*i;
		retList.push_back(cv::Point(xL, 719 - i));
		retList.push_back(cv::Point(xR, 719 - i));
	}
	return retList;
}

vector<cv::Point> combinePoints(deque<vector<double>> p1, deque<vector<double>> p2)
{
	double weights[SMOOTHING_WIN_SIZE];

	vector<double> lsums;
	vector<double> rsums;

	for (int i = 0; i < SMOOTHING_WIN_SIZE; i++)
	{
		weights[i] = 1.0 / (double)SMOOTHING_WIN_SIZE;
	}

	for (int i = 0; i < 3; i++)
	{
		lsums.push_back(0);
		rsums.push_back(0);
	}

	for (int i = 0; i < SMOOTHING_WIN_SIZE; i++)
	{
		lsums[0] += weights[i] * p1[i][0];
		lsums[1] += weights[i] * p1[i][1];
		lsums[2] += weights[i] * p1[i][2];
		rsums[0] += weights[i] * p2[i][0];
		rsums[1] += weights[i] * p2[i][1];
		rsums[2] += weights[i] * p2[i][2];
	}

	return combinePoints(lsums, rsums);
}

vector<double> getSmoothedEq(deque<vector<double>> eq)
{
	vector<double> retVal;
	for (int i = 0; i < 3; i++)
	{
		retVal.push_back(0);
	}

	for (int i = 0; i < SMOOTHING_WIN_SIZE; i++)
	{
		retVal[0] += (double)1 / SMOOTHING_WIN_SIZE * eq[i][0];
		retVal[1] += (double)1 / SMOOTHING_WIN_SIZE * eq[i][1];
		retVal[2] += (double)1 / SMOOTHING_WIN_SIZE * eq[i][2];
	}

	return retVal;
}
//calculates the curvature and the center offset and inserts the values onto final img
void insertCurvatureOffset(Mat & in)
{
	std::ostringstream streamCurv;
	std::ostringstream streamOffset;
	streamCurv << "curvature ";
	

	streamOffset << "offset ";
	streamOffset << std::setprecision(0);
	if (lSmoothReal.size() == SMOOTHING_WIN_SIZE)
	{
		
		vector<double> leftEqRealSmooth = getSmoothedEq(lSmoothReal);
		vector<double> rightEqRealSmoothed = getSmoothedEq(rSmoothReal);
		//offset
		vector<double> leftEqSmooth = getSmoothedEq(lSmooth);
		vector<double> rightEqSmooth = getSmoothedEq(rSmooth);
		double nRows = (double)in.rows;
		double nCols = (double)in.cols;

		double xL = leftEqSmooth[0] + leftEqSmooth[1] * (nRows-1) + leftEqSmooth[2] * pow(nRows-1,2);
		double xR = rightEqSmooth[0] + rightEqSmooth[1] * (nRows-1) + rightEqSmooth[2] * pow(nRows - 1, 2);
		double offset = XM_PER_PIX*abs((nCols/2)-((xL+xR)/2));
	
		
		double left_curverad = ((1 + pow(pow((2 * leftEqRealSmooth[2] * nRows * YM_PER_PIX + leftEqRealSmooth[1]), 2), 1.5) / abs(2 * leftEqRealSmooth[2])));
		double right_curverad = ((1 + pow(pow((2 * rightEqRealSmoothed[2] * nRows * YM_PER_PIX + rightEqRealSmoothed[1]), 2), 1.5) / abs(2 * rightEqRealSmoothed[2])));
		double ave_curverad = (left_curverad + right_curverad) / 2;
	
		streamCurv << std::fixed << std::setprecision(0) <<ave_curverad;
		putText(in, streamCurv.str(), cv::Point(10, 30), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 3, 8);
		
		streamOffset << std::fixed << std::setprecision(2) << offset;
		putText(in, streamOffset.str(), cv::Point(30, 70), FONT_HERSHEY_PLAIN, 2, Scalar::all(255), 3, 8);
	
	}
}

void getLaneIndices(Mat img, int windowStartY, int windowSizeY, int leftWindowStartX, int rightWindowStartX, int windowSizeX, int & leftIdx, int & rightIdx, bool & ltFound, bool &rtFound)
{
	int sums = 0;
	int index_left, index_right;
	int max_left = -1;
	int max_right = -1;
	int windowStartX = 0;
	bool foundLeft = false;
	bool foundRight = true;
	const int minPix = 20;
	vector<int> rtIdxGrp;
	vector<int> ltIdxGrp;

	for (int k = 0; k < 2; k++)
	{
		if (k == 0)
		{
			windowStartX = leftWindowStartX;
		}
		else if (k == 1)
		{
			windowStartX = rightWindowStartX;
		}

		for (int i = windowStartX; i < (windowStartX + windowSizeX); i++)
		{
			sums = 0;

			for (int j = 0; j < (windowSizeY); j++)
			{
				try
				{
					sums += img.at<uchar>(windowStartY - j, i) / 255;
				}
				catch (...)
				{
					int exception = 0;
				}
			}

			if (k == 0)
			{
				if (sums > max_left && sums >= minPix)
				{

					max_left = sums;
					ltIdxGrp.clear();
					ltIdxGrp.push_back(i);
					ltFound = true;
				}
				else if (sums == max_left && sums >= minPix)
				{
					ltIdxGrp.push_back(i);
				}
			}
			else
			{
				if (sums > max_right && sums >= minPix)
				{

					max_right = sums;
					rtIdxGrp.clear();
					rtIdxGrp.push_back(i);
					rtFound = true;
				}
				else if (sums == max_right && sums >= minPix)
				{
					rtIdxGrp.push_back(i);
				}
			}
		}
	}

	if (ltIdxGrp.size() > 0)
	{
		sums = 0;
		for (int i = 0; i < ltIdxGrp.size(); i++)
		{
			sums += ltIdxGrp[i];
		}
		leftIdx = sums / ltIdxGrp.size();
	}

	if (rtIdxGrp.size() > 0)
	{
		sums = 0;
		for (int i = 0; i < rtIdxGrp.size(); i++)
		{
			sums += rtIdxGrp[i];
		}

		rightIdx = sums / rtIdxGrp.size();
	}
}
/*
* returns a vector of points predicted by the polynomial fit equation
*/
vector<cv::Point> detectLanes(Mat in)
{
	const int numWindowsY = 12;

	int margin = 200; //the width in pixels of our search window
	vector<double> leftEqReal;
	vector<double> rightEqReal;
	vector<double> leftEq;
	vector<double> rightEq;
	int winStartLeftX;
	int winStartRightX;
	int windowStartY;
	static int prevRightIdx = 0;
	static int prevLeftIdx = 0;
	bool ltFound = false;
	bool rtFound = false;
	static int ltIdx[numWindowsY];
	static int rtIdx[numWindowsY];
	int curLtIdx = 0;
	int curRtIdx = 0;
	Mat pointsLt = Mat::zeros(in.size(), in.type());
	Mat pointsRt = Mat::zeros(in.size(), in.type());
	int nonZeroPixels = countNonZero(in);
	vector<cv::Point> allPoints;
	Mat tmp;
	//cvtColor(in, tmp, COLOR_GRAY2BGR);

	//the first time this is ever called do a throrough search
	if (prevRightIdx == 0 && prevLeftIdx == 0)
	{
		//first time
		getLaneIndices(in, in.rows - 1, in.rows / 2, 0, in.cols / 2 - 1, in.cols / 2, curLtIdx, curRtIdx, ltFound, rtFound);
		prevLeftIdx = curLtIdx;
		prevRightIdx = curRtIdx;
		//initialize index array with curLtIdx and curRtIdx
		for (int i = 0; i < numWindowsY; i++)
		{
			rtIdx[i] = curRtIdx;
			ltIdx[i] = curLtIdx;
		}
	}

		//loop through each window detecting left and right lines by the columns having the most pixels
		for (int i = 0; i < numWindowsY; i++)
		{
			//for each of our rows calculate the bottom y and the start of our left and right search windows
			winStartLeftX = prevLeftIdx - (margin / 2 + 1);
			winStartRightX = prevRightIdx - (margin / 2 + 1);
			windowStartY = (in.rows - 1) - (i*in.rows / numWindowsY);
			
			if (winStartLeftX <= 0)
				winStartLeftX = 200;

			if (winStartRightX > ((in.cols - 1) - margin))
				winStartRightX = ((in.cols - 1) - margin);
			else if (winStartRightX <= 0)
				winStartRightX = winStartLeftX + 700;

		
			ltFound = false;
			rtFound = false;

			getLaneIndices(in, windowStartY, in.rows / numWindowsY, winStartLeftX, winStartRightX, margin, curLtIdx, curRtIdx, ltFound, rtFound);

			int tmpRtIdx = rtIdx[i];
			int tmpLtIdx = ltIdx[i];
			
			prevRightIdx = curRtIdx;
			prevLeftIdx = curLtIdx;
			if (rtFound == false)
			{
				prevRightIdx = tmpRtIdx;
			}
			else
			{
				rtIdx[i] = curRtIdx;
			}

			if (ltFound == false)
			{
				prevLeftIdx = tmpLtIdx;
			}
			else
			{
				ltIdx[i] = curLtIdx;
			}

			//cv::Point bottomLeft = (winStartLeftX, (i*in.rows / numWindowsY));
			//cv::Point topRight = (winStartLeftX + margin, (i*in.rows / numWindowsY) + in.rows / numWindowsY);
			//rectangle(tmp, cv::Point(winStartLeftX, 719 - (i*in.rows / numWindowsY)), cv::Point(winStartLeftX + margin, 719 - (i*in.rows / numWindowsY) + in.rows / numWindowsY), Scalar(110, 220, 0), 1, 8, 0);
			//rectangle(tmp, cv::Point(winStartRightX, 719 - (i*in.rows / numWindowsY)), cv::Point(winStartRightX + margin, 719 - (i*in.rows / numWindowsY) + in.rows / numWindowsY), Scalar(110, 220, 0), 1, 8, 0);

			generateFakeData(pointsLt, pointsRt, in.rows / numWindowsY, windowStartY, ltIdx[i], rtIdx[i]);
		}

		prevLeftIdx = ltIdx[0];
		prevRightIdx = rtIdx[0];

		transpose(pointsLt, pointsLt);
		flip(pointsLt, pointsLt, 1);
		transpose(pointsRt, pointsRt);
		flip(pointsRt, pointsRt, 1);

		leftEqReal = mPolyfit(pointsLt, true);
		rightEqReal = mPolyfit(pointsRt, true);
		leftEq = mPolyfit(pointsLt, false);
		rightEq = mPolyfit(pointsRt, false);

		//imshow("detection", tmp);

		if (lSmooth.size() >= SMOOTHING_WIN_SIZE)
		{
			lSmooth.pop_back();
			lSmoothReal.pop_back();
		}

		if (rSmooth.size() >= SMOOTHING_WIN_SIZE)
		{
			rSmooth.pop_back();
			rSmoothReal.pop_back();
		}

		lSmooth.push_front(leftEq);
		rSmooth.push_front(rightEq);
		lSmoothReal.push_front(leftEqReal);
		rSmoothReal.push_front(rightEqReal);


		allPoints.clear();
		if ((lSmooth.size() == SMOOTHING_WIN_SIZE) && (lSmooth.size() == SMOOTHING_WIN_SIZE))
		{
			allPoints = combinePoints(lSmooth, rSmooth);
		}
		else
		{
			allPoints = combinePoints(leftEq, rightEq);
		}

		return allPoints;
}

int main()
{
	VideoCapture camStream;
	VideoWriter video;
	Mat camFrame,initFrame,filteredFrame,undistortedFrame,perspectiveFrame,warpedFrame,binaryWarpedFrame, colorWarpedFrame, processedFrame;
	Mat intrinsic, distCoeff;
	Mat M, Minv;
	int metallicaRocks = 0;
	vector<cv::Point> lanePoints;


	//get the calibration matrixes
	calibrate(intrinsic,distCoeff,false); 
	cout << "Finished Calibration\n";

	//get the 3 x 3 matrix of the predefined perspective transform
	initWarpImg(M, Minv);

	camStream.open(".\\project_video.mp4");
	bool notFinished = camStream.read(camFrame);

#if WRITE_VIDEO_ENABLED == 1
	if (video.isOpened() == false)
	{
		int frame_width = camStream.get(CV_CAP_PROP_FRAME_WIDTH);
		int frame_height = camStream.get(CV_CAP_PROP_FRAME_HEIGHT);
		video.open("./project_video_solution.mp4", CV_FOURCC('M', 'P', '4', 'V'), 22, cv::Size(frame_width, frame_height), true);
	}
#endif

	//this is the main loop pipeline!
	while (notFinished == true)
	{
		//init frame is pristine and not manipulated
		initFrame = camFrame.clone();

		//undistort current frame 
		undistort(camFrame, undistortedFrame, intrinsic, distCoeff);
	
		//apply perspective transform
		warpPerspective(undistortedFrame, warpedFrame, M, undistortedFrame.size());
		
		//apply filters and thresholding on warped image
		applyFilters(warpedFrame, binaryWarpedFrame);
		
		//perform lane detection algorithm. Lane points contains the points 
		//corresponding to the estimated equations for the left and right lanes
		lanePoints = detectLanes(binaryWarpedFrame);
		
		const cv::Point* elementPoints[1] = { &lanePoints[0] };
		int numberOfPoints = (int)lanePoints.size();

		colorWarpedFrame = Mat::zeros(binaryWarpedFrame.size(), CV_8UC3);

		//color the space in between the 2 detected lanes
		fillPoly(colorWarpedFrame, elementPoints, &numberOfPoints, 1, Scalar(0, 255, 0), 8);

		//unwarp image with highlighted lane back to original 
		warpPerspective(colorWarpedFrame, colorWarpedFrame, Minv, colorWarpedFrame.size());

		//overlay highlighted lane on original image
		addWeighted(initFrame, 1, colorWarpedFrame, .3, 0, processedFrame);

		//calculate and insert curvature and offset information 
		insertCurvatureOffset(processedFrame);

		imshow("processed frame", processedFrame);
		waitKey(1);
		
		//read the next frame
		notFinished = camStream.read(camFrame);

#if WRITE_VIDEO_ENABLED == 1
		video.write(processedFrame);
#endif
	}

#if WRITE_VIDEO_ENABLED == 1
	video.release();
#endif

	cout << "Press Any Key to close console...";
	cout.flush();
	cin.get();

	cvDestroyAllWindows();
    return 0;
}

