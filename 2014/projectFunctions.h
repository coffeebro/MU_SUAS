//
//  projectFunctions.h
//  capstone_functions
//
//  Created by Aaron Scantlin on 12/7/14.
//  Copyright (c) 2014 Aaron Scantlin. All rights reserved.
//

#ifndef __capstone_functions__projectFunctions__
#define __capstone_functions__projectFunctions__

#include <stdio.h>
#include "projectHeaders.h"

using namespace std;
using namespace cv;

//Function headers
Mat CreateMatFromImage ( string fullPathToImage );
Mat CreateThreshold ( Mat image, double lowH, double lowS, double lowV, double highH, double highS, double highV, bool invert );
vector< Mat > FindCandidateTargets ( Mat original, Mat image, float minArea, float maxArea, int color, int *counter, int *x, int *y );
String DetermineShape ( Mat image );
char* DetectTargetColor( Mat image, int *b, int *g, int *r );
char* DetectCharacterColor( Mat image, int *b, int *g, int *r );
char* GetColorName( int red, int green, int blue );
void RotateImage( Mat& src, double angle, Mat& dst );
Mat BinaryImage ( Mat image );
String Identify( Mat src, Mat refineMe );
vector< Mat > CreateClustersFromMat ( Mat image, int clusterCount );
Mat RemoveColorsFromImage ( Mat image, map <String, bool> colors );
void ErrorDialogue ( string error );
vector< Mat > GatherResults(const cv::Mat& labels, const cv::Mat& centers, int height, int width);
string LowerLetter (String convertMe);
double CalcTime (time_t start, time_t end, string format);

#endif /* defined(__capstone_functions__projectFunctions__) */