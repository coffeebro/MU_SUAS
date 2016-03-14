//
//  projectFunctions.cpp
//  capstone_functions
//
//  Created by Aaron Scantlin on 12/7/14.
//  Copyright (c) 2014 Aaron Scantlin. All rights reserved.
//

#include "projectFunctions.h"

using namespace tesseract;

Mat CreateMatFromImage ( string fullPathToImage ) {
    /*
     Input: string that is an explicit path to an image file
     
     Output: a BGR Mat element with the same dimensions as the supplied image file
     */
    
    //Read an image from file
    Mat originalImage = imread( fullPathToImage );
    
    //Call error function if no image is loaded into the Mat
    if ( originalImage.empty() == true ) {
        ErrorDialogue( "Could not find image at location " + fullPathToImage );
        exit( -1 );
    }
    
    return originalImage;
}

Mat CreateThreshold ( Mat image, double lowH, double lowS, double lowV, double highH, double highS, double highV, bool invert ) {
    /*
     Input: a Mat element, as well as lower and upper bounds for hue, saturation and value
     
     Output: a binary Mat with the same dimensions as the supplied image file scaled to
     the supplied hue, saturation and value values
     */
    
    Mat hsv;
    
    //Convert BGR image to HSV
    cvtColor( image, hsv, CV_BGR2HSV );
    
    //Call error function if no image is loaded into the Mat
    if ( hsv.empty() == true ) {
        ErrorDialogue( "Could not convert image to HSV (are you passing this function a BGR image?)" );
        exit( -1 );
    }
    
    //Re-render HSV image as binary within specified range of values
    inRange( hsv, Scalar( lowH, lowS, lowV ), Scalar( highH, highS, highV ), hsv );
    
    if (invert == true) {
        bitwise_not( hsv, hsv );
    }
    
    return hsv;
}

vector< Mat> FindCandidateTargets ( Mat original, Mat image, float minArea, float maxArea, int color, int *counter, int *x, int *y ) {
    /*
     Input: a Mat element containing the original image, a Mat element containing the thresholded
     original image, a minimum pixel area, and a maximum pixel area
     
     Output: an array of vectors containing the Mat elements of the cropped candidate targets
     */
    
    //Look for any blobs with a pixel area between minArea and maxArea
    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;
    params.filterByColor = true;
    params.minArea = minArea;
    params.maxArea = maxArea;
    params.blobColor = color;
    
    SimpleBlobDetector blobDetector( params );
    vector< KeyPoint > keyPoints;
    vector< Mat > candidates;
    Mat crop;
    Mat keypts;
    Rect roi;
    *counter = 0;
    
    //Detect the blobs with the above parameters and store their locations in keyPoints
    blobDetector.detect( image, keyPoints );
    
    cout << "\n";
    
    //Determine a bounding box for each blob and crop it out of the original image
    for ( int i = 0; i < keyPoints.size(); i++ ) {
        
        //Compute bounding box
        roi.x = std::max( 0, ( int )( keyPoints[i].pt.x - keyPoints[i].size * 2 ) );
        roi.y = std::max( 0, ( int )( keyPoints[i].pt.y - keyPoints[i].size * 2 ) );
        roi.width = ( int )( keyPoints[i].size * 4 );
        roi.height = ( int )( keyPoints[i].size * 4 );
        
        //todo change to middle of target
        printf( "Candidate target located at (x:%d, y:%d)\n", ( int )keyPoints[i].pt.x, ( int )keyPoints[i].pt.y );
        
        /*
         Take the bounding box measurements calculated above and remove that section from the original image,
         creating a smaller, cropped image of the candidate target.. then push that onto an array of candidates
         */
        try {
            crop = original(roi);
        }
        catch ( ... ) {
            printf( "ROI would break the universe, moving on..\n\n" );
            continue;
        }
        
        //keep track of the X and Y coordinates of the target
        x[*counter] = ( int )keyPoints[i].pt.x;
        y[*counter] = ( int )keyPoints[i].pt.y;
        candidates.push_back( crop );
        ( *counter )++;
    }
    
    cout << "\n";
    *counter = 0;
    
    return candidates;
}

vector< Mat > CreateClustersFromMat ( Mat image, int clusterCount ) {
    /*
     Input: A Mat element to cluster, number of clusters to create
     
     Output: An array of Mat elements, each containing a single cluster
     */
    
    //This variable is kinda BS since we use TERMCRIT_ITER to determine
    //how many attempts will be made
    int attempts = 5;
    
    //Format the image so that k means can work with it
    Mat reshapedImage = image.reshape( 1, image.cols * image.rows );
    assert( reshapedImage.type() == CV_8UC1 );
    Mat reshapedImage32f;
    reshapedImage.convertTo( reshapedImage32f, CV_32FC1, 1.0 / 255.0 );
    assert( reshapedImage32f.type() == CV_32FC1 );
    
    cout << "Creating clusters from image with k means, please wait..\n";
    
    Mat labels, centers;
    vector< Mat > clusters;
    
    //kmeans will run up to 1000 times or until the abs of the centers change insufficiently
    kmeans( reshapedImage32f, clusterCount, labels, TermCriteria( CV_TERMCRIT_ITER, 1000, 1 ), attempts, KMEANS_PP_CENTERS, centers );
    
    //Load each individual cluster into its own Mat element
    clusters = GatherResults( labels, centers, image.rows, image.cols );
    
    return clusters;
}

String DetermineShape ( Mat image ) {
    /*
     Input: a cleaned up binary Mat element containing the cropped photo of a target
     
     Output: a String representing the shape of the target
     */
    
    string shape = "UNKNOWN SHAPE";
    vector< vector < Point > > contours;
    vector< Vec4i > hierarchy;
    
    //Find and draw the contours of the target
    try {
        findContours( image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
        drawContours( image, contours, -1, Scalar( 255, 255, 255 ), -1 );
    }
    catch ( ... ) {
        cout << "Unable to find/draw contours" << endl;
        return "UNKNOWN SHAPE";
    }
    
    //Blur the image to remove any tiny inconsistencies from image cleanup
    blur( image, image, Size( 5, 5 ) );
    
    //Use approxPolyDP to detect how many sides the target has
    vector< Point > contoursOUT;
    try {
        approxPolyDP( Mat( contours[0] ), contoursOUT, arcLength( Mat( contours[0] ), true )*0.02, true );
    }
    catch ( ... ) {
        cout << "Function approxPolyDP failed" << endl;
        return "UNKNOWN SHAPE";
    }
    
    //Classify target shape based on number of sides
    if ( contoursOUT.size() == 3 ) {
        shape = "triangle";
    }
    else if ( contoursOUT.size() == 4 ) {
        shape = "4-gon";
    }
    else if ( contoursOUT.size() == 5 ) {
        shape = "pentagon";
    }
    else if ( contoursOUT.size() == 6 ) {
        shape = "hexagon";
    }
    else if ( contoursOUT.size() > 6 ) {
        shape = "circle";
    }
    
    return shape;
}

char* DetectTargetColor( Mat image, int *b, int *g, int *r ) {
    /*
     Input: a Mat element containing the isolated image of the target (no background)
     
     Output: a pointer to a Char containing the background color of the target
     */
    
    Vec3b *pixel = new Vec3b [image.rows * image.cols];
    Vec3b test;
    int max = 0, index = 0, count = 0;
    
    cout << "Detecting target color, please wait..";
    
    /*
     For each row and column, determine the BGR values of a pixel.
     If it's black, skip over it; otherwise, store the pixel in an array.
     */
    for ( int i = 0; i < image.rows; i++ ) {
        for ( int j = 0; j < image.cols; j++ ) {
            test = image.at< Vec3b >( i, j );
            
            if ( test.val[0] == 0 && test.val[1] == 0 && test.val[2] == 0 ) {
                continue;
            }
            else {
                pixel[index] = image.at< Vec3b >( i, j );
                index++;
            }
        }
    }
    
    cout << ".";
    
    /*
     For every pixel sampled in a row, look at it.
     For each of those row pixels, compare it to every column pixel.
     If the values of the two pixels are the same, increase the count.
     Then, for each row pixel sampled, if count is greater than max, assign max = count.
     This will give us the BGR values of the row pixel that matches the most column pixels.
     */
    for ( int i = 0; i < index; i++ ) {
        count = 0;
        
        for ( int j = 0; j < index; j++ ) {
            if ( pixel[i].val[0] == pixel[j].val[0] && pixel[i].val[1] == pixel[j].val[1] && pixel[i].val[2] == pixel[j].val[2] ) {
                count++;
            }
        }
        
        if ( count > max ) {
            max = count;
        }
    }
    
    cout << ".";
    
    /*
     Perform the same operation as above.
     This time, when count is equal to max, we know we have the most common BGR values in the supplied Mat element.
     */
    for ( int i = 0; i < index; i++ ) {
        int count = 0;
        
        for ( int j = 0; j < index; j++ ) {
            if ( pixel[i].val[0] == pixel[j].val[0] && pixel[i].val[1] == pixel[j].val[1] && pixel[i].val[2] == pixel[j].val[2] ) {
                count++;
            }
        }
        
        if ( count == max ) {
            *b = pixel[i].val[0];
            *g = pixel[i].val[1];
            *r = pixel[i].val[2];
        }
    }
    
    cout << ".";
    
    delete[] pixel;
    
    //Grab the name of the color based on the most common BGR values that we just found.
    char* color = GetColorName( *r, *g, *b );
    
    cout << "\n";
    
    return color;
}

char* DetectCharacterColor( Mat image, int *b, int *g, int *r ) {
    /*
     Input: a Mat element containing the isolated image of the target (no background)
     
     Output: a pointer to a Char containing the color of the inner target
     */
    
    Vec3b *pixel = new Vec3b [image.rows * image.cols];
    Vec3b test;
    int index = 0, count = 0, max = 0;
    
    cout << "Detecting character color, please wait.." << endl;
    
    /*
     For each row and column, determine the BGR values of a pixel.
     If it's black, skip over it.
     If it's close enough to the most prominent color in the picture, change it to black.
     Otherwise, store the pixel in an array.
     */
    for ( int i = 0; i < image.rows; i++ ) {
        for ( int j = 0; j < image.cols; j++ ) {
            test = image.at< Vec3b >( i, j );
            
            if ( test.val[0] == 0 && test.val[1] == 0 && test.val[2] == 0 ) {
                continue;
            }
            else {
                int distance = abs( *r - test.val[2]) + abs(*g - test.val[1]) + abs(*b - test.val[0] );
                if ( distance < 80 ) {
                    image.at< Vec3b >( i, j ).val[0] = 0;
                    image.at< Vec3b >( i, j ).val[1] = 0;
                    image.at< Vec3b >( i, j ).val[2] = 0;
                }
                else {
                    pixel[index] = image.at< Vec3b >( i, j );
                }
                index++;
            }
            
        }
    }
    
    pixel = new Vec3b [image.rows * image.cols];
    test = Vec3b();
    index = 0;
    
    /*
     For each row and column, determine the BGR values of a pixel.
     If it's black, skip over it; otherwise, store the pixel in an array.
     */
    for ( int i = 0; i < image.rows; i++ ) {
        for ( int j = 0; j < image.cols; j++ ) {
            test = image.at< Vec3b >( i, j );
            
            if ( test.val[0] == 0 && test.val[1] == 0 && test.val[2] == 0 ) {
                continue;
            }
            else {
                pixel[index] = image.at< Vec3b >( i, j );
                index++;
            }
        }
    }
    
    /*
     For every pixel sampled in a row, look at it.
     For each of those row pixels, compare it to every column pixel.
     If the values of the two pixels are the same, increase the count.
     Then, for each row pixel sampled, if count is greater than max, assign max = count.
     This will give us the BGR values of the row pixel that matches the most column pixels.
     */
    for ( int i = 0; i < index; i++ ) {
        count = 0;
        
        for ( int j = 0; j < index; j++ ) {
            if ( pixel[i].val[0] == pixel[j].val[0] && pixel[i].val[1] == pixel[j].val[1] && pixel[i].val[2] == pixel[j].val[2] ) {
                count++;
            }
        }
        
        if ( count > max ) {
            max = count;
        }
    }
    
    /*
     Perform the same operation as above.
     This time, when count is equal to max, we know we have the second most common BGR values in the supplied Mat element.
     */
    for ( int i = 0; i < index; i++ ) {
        count = 0;
        
        for ( int j = 0; j < index; j++ ) {
            if ( pixel[i].val[0] == pixel[j].val[0] && pixel[i].val[1] == pixel[j].val[1] && pixel[i].val[2] == pixel[j].val[2] ) {
                count++;
            }
        }
        
        if ( count == max ) {
            *b = pixel[i].val[0];
            *g = pixel[i].val[1];
            *r = pixel[i].val[2];
        }
    }
    
    delete[] pixel;
    
    //Grab the name of the color based on the most common BGR values that we just found.
    char* color = GetColorName( *r, *g, *b );
    
    return color;
}

char* GetColorName ( int red, int green, int blue ) {
    /*
     Input: the R, G, and B values of the color we want to label
     
     Output: the name of a color
     */
    
    /*
     Adam I have no idea what's going on with the math here, help a brother out. B)
     */
    char* colorNames[13] = { "black", "red", "orange", "yellow", "green", "cyan", "blue", "blue", "purple", "magenta", "pink", "grey", "white" };
    int colorRed[13] = { 0, 255, 255, 255, 0, 51, 51, 0, 178, 255, 255, 160, 255 };
    int colorGreen[13] = { 0, 51, 128, 255, 255, 255, 153, 0, 102, 51, 153, 160, 255 };
    int colorBlue[13] = { 0, 51, 0, 51, 0, 255, 255, 255, 255, 255, 204, 160, 255 };
    
    int diff[13] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    int i;
    int minDiff = 765;
    int minIndex = 0;
    
    for ( i = 0; i < 13; i++ ){
        diff[i] = abs( red - colorRed[i] ) + abs( green - colorGreen[i] ) + abs( blue - colorBlue[i] );
    }
    
    for ( i = 0; i < 13; i++ ){
        if ( diff[i] < minDiff ){
            minDiff = diff[i];
            minIndex = i;
        }
    }
    
    return colorNames[minIndex];
}

void RotateImage( Mat& src, double angle, Mat& dst ) {
    int len = max( src.cols, src.rows );
    Point2f pt( len / 2.0, len / 2.0 );
    Mat r = getRotationMatrix2D( pt, angle, 1.0 );
    
    warpAffine( src, dst, r, Size( len, len ), INTER_CUBIC, BORDER_CONSTANT, Scalar( 255, 255, 255 ) ); //the scalar is what does the white background
}

Mat BinaryImage ( Mat image ) {
    /*
     Input: a Mat element populated with data
     
     Output: a binary Mat element of the input data
     */
    
    cvtColor( image, image, CV_BGR2GRAY );
    
    Mat binary( image.size(), image.type() );
    
    //Threshold the input Mat element
    threshold( image, binary, 0, 255, THRESH_BINARY );
    
    return binary;
}

string Identify( Mat src, Mat refineMe ) {
    /*
     * Function to identify a capital letter
     * Input: a blank Mat element and a Mat containing the inner target to be refined into a binary image
     * Output: string with letter and confidence
     */
    
    Vec3b test;
    
    for ( int i = 0; i < refineMe.rows; i++ ) {
        for ( int j = 0; j < refineMe.cols; j++ ) {
            test = refineMe.at< Vec3b >( i, j );
            
            if ( test.val[0] == 255 && test.val[1] == 255 && test.val[2] == 255 ) {
                continue;
            }
            else {
                test.val[0] = 0;
                test.val[1] = 0;
                test.val[2] = 0;
                
                refineMe.at<Vec3b>( i, j ) = test;
            }
        }
    }
    
    src = refineMe;
    
    //variable declaration
    int i, j, n, threshold, avg;
    Mat cannySrc, grayCannySrc;
    Mat perpendicularImages[4]; //array to store the 4 rotated Mat images
    vector< Vec2f > lines; //stores the lines from Hough transform (lowercase vector is an OpenCV thing, not to be confused with std::Vector)
    Vector< double > angles; //vector for storing the raw angles
    angles.reserve( 100 ); //give it some storage space, so we don't have to resize later (vectors reallocate automatically, but it costs CPU time)
    Vector< double > avgAngles;
    avgAngles.reserve( 10 );
    Vector< double > adjustedAngles; //vector for the averaged angle values
    adjustedAngles.reserve( 10 );//give it some storage space, so we don't have to resize later
    //vectors for output, we will reserve their size when it is known later on
    Vector< string > outputs;
    Vector< float > confs;
    TessBaseAPI tess; //tesseract handle
    ResultIterator* ri; //used for working with tess output
    PageIteratorLevel level = RIL_WORD;
    float conf, finalConf; //confidence level of OCR result
    char* output;
    string finalOutput;
    
    try
    {
        
        //perform a canny edge detection and convert the result to grayscale
        Canny( src, cannySrc, 50, 200, 3 );
        cvtColor( cannySrc, grayCannySrc, CV_GRAY2BGR );
        
        threshold = 40;
        
        //start with high threshold and move to lower threshold until around 3 lines are found (arbitrary, but seems to work well enough from my data)
        while ( avgAngles.size() < 3 ) {
            
            //make sure lines, angles, and avgAngles are empty before re-running
            lines.clear();
            angles.clear();
            avgAngles.clear();
            
            //do the hough transform to find the lines
            HoughLines(  cannySrc, lines, 1, CV_PI / 180, threshold, 0, 0);
            threshold--;
            
            //only do the rest of this if we got any lines
            if ( !lines.empty() ) {
                //calculate the lines and angles
                for ( size_t i = 0; i < lines.size(); i++ ) {
                    float rho = lines[i][0], theta = lines[i][1];
                    Point pt1, pt2;
                    double a = cos( theta ), b = sin( theta );
                    double x0 = a*rho, y0 = b*rho;
                    pt1.x = cvRound( x0 + 1000 * ( -b ) );
                    pt1.y = cvRound( y0 + 1000 * ( a ) );
                    pt2.x = cvRound( x0 - 1000 * ( -b ) );
                    pt2.y = cvRound( y0 - 1000 * ( a ) );
                    
                    //record the angle of the line (converted to degrees)
                    angles.push_back( atan2( pt1.y - pt2.y, pt1.x - pt2.x ) * ( 180.0 / 3.1459 ) );
                }
                //sort the vector in numeric order
                sort( angles.begin(), angles.end() );
                
                j = 0;
                avg = angles[0];
                
                // average out the multiple angle values into discrete clumps which represent the "true" angles
                for ( n = 0; n < angles.size(); n++ ) {
                    
                    //check if we are on the same angle, or a new angle
                    if ( angles[n] < avg + 5 && angles[n] > avg - 5 ) {
                        //take the rolling avg
                        avg = ( avg + angles[n] ) / 2.0;
                        //if we are at the end of the list, grab the avg
                        if ( n + 1 == angles.size() ) {
                            avgAngles.push_back( avg );
                        }
                    }
                    else {
                        //store the old avg, as we have finished an angle
                        avgAngles.push_back( avg );
                        avg = angles[n]; //set it to the next angle
                    }
                }
            }
        }
        
        
        //reserve the size of the output vectors (number of avg angles times 4, since there will be 4 images to analyze)
        outputs.reserve( avgAngles.size() * 4 );
        confs.reserve( avgAngles.size() * 4 );
        
        //now let's do the OCR
        //Initialize the Tesseract API
        tess.Init( NULL, "eng", OEM_DEFAULT );
        tess.SetPageSegMode( PSM_SINGLE_CHAR ); //tell tesseract to look for a single char
        
        //init the output and conf final values
        finalOutput = "";
        finalConf = 0.0;
        
        //loop through every averaged angle we found
        for ( i = 0; i < avgAngles.size(); i++ ) {
            //do the rotation based on the angle
            RotateImage( src, avgAngles[i], perpendicularImages[0] ); //rotate the src image by the angle
            RotateImage( src, avgAngles[i] + 90, perpendicularImages[1] ); //now add 90 degrees for each orientation of the rotated image
            RotateImage( src, avgAngles[i] + 180, perpendicularImages[2] );
            RotateImage( src, avgAngles[i] + 260, perpendicularImages[3] );
            
            //do for each of the perp. images
            for ( j = 0; j < 4; j++ ) {
                //this call converts from Mat to PIX format then uses this as the image to be OCR'ed
                tess.SetImage( ( uchar* ) perpendicularImages[j].data, perpendicularImages[j].size().width, perpendicularImages[j].size().height, perpendicularImages[j].channels(), perpendicularImages[j].step1() );
                tess.Recognize( 0 );
                //iterate through the results
                ri = tess.GetIterator();
                if ( ri != 0 ) {
                    do {
                        //get the output
                        output = ri->GetUTF8Text( level );
                        //get the confidence
                        conf = ri->Confidence( level );
                        
                        //check for new best confidence
                        if ( conf > finalConf ) {
                            string temp( output );
                            
                            //also discard results that are not numeric and capital
                            if ( !temp.empty() && temp != "" ){
                                //sometimes tesseract will return a weird string that comes out as a negative value, which will crash the program. So, skip it if it does that
                                if ( ( int ) temp.at( 0 ) > 0 ) {
                                    if ( !isupper( ( int )temp.at( 0 ) ) ) continue;
                                }
                                else {
                                    continue;
                                }
                            }
                            
                            finalOutput = temp;
                            finalConf = conf;
                            //cout << "New high score!\n";
                            //cout << "Letter: " << finalOutput << "\t Confidence: " << finalConf << "\n"; //for some really weird reason, this string only works with cout and not printf...
                        }
                        
                    } while ( ri->Next( level ) );
                }
            }
        }
    }
    catch ( ... )
    {
        cout << "An exception occurred" << endl;
        return "BAD";
    }
    
    //the last check in this if statement is kind of cheating, but this
    //function  practically thinks anything that isn't a letter is
    //the letter I..
    if ( finalConf < 71 || finalOutput == "I" )
        return "BAD";
    
    ostringstream ss;
    ss << finalConf;
    
    return finalOutput + " " + ss.str();
}

Mat RemoveColorsFromImage ( Mat image, map < String, bool > colors ) {
    /*
     Input: A Mat element and a map of colors to remove from the Mat
     
     Output: The Mat element minus the specified colors
     */
    
    Mat finishedProduct, originalThresh, noGreen, noBrown, noGray;
    int threshSeq = 0;
    
    if ( colors["Green"] == true ) {
        originalThresh = CreateThreshold( image, 27.5, 0, 0, 80, 255, 255, true );
        image.copyTo( noGreen, originalThresh );
        threshSeq = 1;
    }
    
    if ( colors["Brown"] == true && colors["Green"] == true && threshSeq == 1 ) {
        originalThresh = CreateThreshold( image, 15, 0, 0, 25, 255, 255, true );
        noGreen.copyTo( noBrown, originalThresh );
        threshSeq = 2;
    }
    else if ( colors["Brown"] == true && threshSeq == 0 ) {
        originalThresh = CreateThreshold( image, 15, 0, 0, 25, 255, 255, true );
        image.copyTo( noBrown, originalThresh );
        threshSeq = 1;
    }
    
    if ( colors["Gray"] == true && colors["Green"] == true && colors["Brown"] == true && threshSeq == 2 ) {
        originalThresh = CreateThreshold( image, 0, 0, 51, 180, 25.5, 196.35, true );
        noBrown.copyTo( noGray, originalThresh );
        threshSeq = 3;
    }
    else if ( colors["Gray"] == true && colors["Green"] == true && colors["Brown"] == false && threshSeq == 1 ) {
        originalThresh = CreateThreshold( image, 0, 0, 51, 180, 25.5, 196.35, true );
        noGreen.copyTo( noGray, originalThresh );
        threshSeq = 2;
    }
    else if ( colors["Gray"] == true && colors["Green"] == false && colors["Brown"] == true && threshSeq == 1 ) {
        originalThresh = CreateThreshold( image, 0, 0, 51, 180, 25.5, 196.35, true );
        noBrown.copyTo( noGray, originalThresh );
        threshSeq = 2;
    }
    else if ( colors["Gray"] == true && threshSeq == 0 ) {
        originalThresh = CreateThreshold( image, 0, 0, 51, 180, 25.5, 196.35, true );
        image.copyTo( noGray, originalThresh );
        threshSeq = 1;
    }
    
    if ( !noGray.empty() ) {
        return noGray;
    }
    else if ( !noBrown.empty() ) {
        return noBrown;
    }
    else if ( !noGreen.empty() ) {
        return noGreen;
    }
    else {
        return image;
    }
}

void ErrorDialogue ( string error ) {
    /*
     Input: a String containing an error message
     
     Output: none
     */
    
    cout << error << "\n";
    cout << "Press any key to continue..";
    cin.ignore();
}

vector< Mat > GatherResults( const cv::Mat& labels, const cv::Mat& centers, int height, int width ) {
    /*
     Input: The labels output from the kmeans function, the centers output from the kmeans function, the height of the image, and the width of the image
     
     Output: An array of 3 Mat elements each containing a single cluster
     */
    
    Vec3b white;
    white.val[0] = 255;
    white.val[1] = 255;
    white.val[2] = 255;
    
    Vec3b vals1 = 0;
    Vec3b vals2 = 0;
    Vec3b vals3 = 0;
    Vec3b vals4 = 0;
    Vec3b vals5 = 0;
    
    std::cout << "===\n";
    std::cout << "labels: " << labels.rows << " " << labels.cols << std::endl;
    std::cout << "centers: " << centers.rows << " " << centers.cols << std::endl;
    assert(labels.type() == CV_32SC1);
    assert(centers.type() == CV_32FC1);
    
    cv::Mat rgb_image(height, width, CV_8UC3);
    cv::MatIterator_<cv::Vec3b> rgb_first = rgb_image.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> rgb_last = rgb_image.end<cv::Vec3b>();
    cv::MatConstIterator_<int> label_first = labels.begin<int>();
    
    cv::Mat cluster1(height, width, CV_8UC3);
    cv::Mat cluster2(height, width, CV_8UC3);
    cv::Mat cluster3(height, width, CV_8UC3);
    cv::Mat cluster4(height, width, CV_8UC3);
    cv::Mat cluster5(height, width, CV_8UC3);
    
    cv::MatIterator_<cv::Vec3b> c1_first = cluster1.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> c2_first = cluster2.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> c3_first = cluster3.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> c4_first = cluster4.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> c5_first = cluster5.begin<cv::Vec3b>();
    
    cv::Mat centers_u8;
    centers.convertTo(centers_u8, CV_8UC1, 255.0);
    cv::Mat centers_u8c3 = centers_u8.reshape(3);
    
    while ( rgb_first != rgb_last ) {
        const cv::Vec3b& valTest = centers_u8c3.ptr<cv::Vec3b>(*label_first)[0];
        
        if (rgb_first == rgb_image.begin<cv::Vec3b>()) {
            vals1 = valTest;
        }
        else {
            if (valTest != vals1) {
                if (vals2 == vals3) {
                    vals2 = valTest;
                }
                
                else if (valTest != vals2) {
                    if (vals3 == vals4) {
                        vals3 = valTest;
                    }
                    
                    else if (valTest != vals3) {
                        if (vals4 == vals5) {
                            vals4 = valTest;
                        }
                        
                        else if (valTest != vals4) {
                            vals5 = valTest;
                            break;
                        }
                    }
                }
            }
        }
        
        ++rgb_first;
        ++label_first;
    }
    
    rgb_first = rgb_image.begin<cv::Vec3b>();
    label_first = labels.begin<int>();
    
    while ( rgb_first != rgb_last ) {
        const cv::Vec3b& rgb = centers_u8c3.ptr<cv::Vec3b>(*label_first)[0];
        
        if (rgb == vals1) {
            *c1_first = rgb;
            ++c1_first;
        }
        else {
            *c1_first = white;
            ++c1_first;
        }
        
        if (rgb == vals2) {
            *c2_first = rgb;
            ++c2_first;
        }
        else {
            *c2_first = white;
            ++c2_first;
        }
        
        if (rgb == vals3) {
            *c3_first = rgb;
            ++c3_first;
        }
        else {
            *c3_first = white;
            ++c3_first;
        }
        
        if (rgb == vals4) {
            *c4_first = rgb;
            ++c4_first;
        }
        else {
            *c4_first = white;
            ++c4_first;
        }
        
        if (rgb == vals5) {
            *c5_first = rgb;
            ++c5_first;
        }
        else {
            *c5_first = white;
            ++c5_first;
        }
        
        ++rgb_first;
        ++label_first;
    }
    
    vector< Mat > clusters;
    if (!cluster5.empty())
    clusters.push_back(cluster5);
    if (!cluster4.empty())
    clusters.push_back(cluster4);
    if (!cluster3.empty())
    clusters.push_back(cluster3);
    if (!cluster2.empty())
    clusters.push_back(cluster2);
    if (!cluster1.empty())
    clusters.push_back(cluster1);
    
    return clusters;
}

string LowerLetter ( string convertMe ) {
    string theLower = "?";
    string uLetters[26] = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
        "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
    string lLetters[26] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
    
    for (int i = 0; i < 26; i++) {
        if (convertMe == uLetters[i])
            return lLetters[i];
    }
    
    return theLower;
}

double CalcTime ( time_t start, time_t end, string format ) {
    double theTime = 0.0;
    
    if (format == "sec") {
        theTime = difftime(end, start);
    }
    else if (format == "min") {
        theTime = (difftime(end, start))/60;
    }
    else if (format == "hrs") {
        theTime = ((difftime(end, start))/60)/60;
    }
    
    return theTime;
}