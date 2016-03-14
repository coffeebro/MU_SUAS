/*
 2014 Capstone Project - Team DFSH
 Target Recognition and Classification Software
 Authors:
 Chris Dopuch, Adam Faszl, Ryan Haslag, Aaron Scantlin
 */

#include "projectHeaders.h"
#include "projectFunctions.h"

using namespace cv;
using namespace std;
using namespace tesseract;

//Global variables
const char *dir = "/Users/Aaron/Pictures/output"; //Edit this line to suit your machine/filename
const char *jsonOut = "/Users/Aaron/Desktop/FinalOutput/output.json"; //Edit this line to suit your machine/filename
const char *finalOut = "/Users/Aaron/Desktop/FinalOutput/"; //Edit this line to suit your machine/filename
const char *candidateDir = "/Users/Aaron/Desktop/Output/candidate"; //Edit this line to suit your machine/filename

//Verbose debugging output, 0=off, 1=on
int verbose = 1;

int main ( void ) {
    int finalCount = 0, bFinal = 0, gFinal = 0, rFinal = 0, writeCandidateCounter = 0, readCandidateCounter = 0;
    int posCounter = 0, xPos[XY_BUFFER] = {0}, yPos[XY_BUFFER] = {0};
    ofstream jsonOutput;
    ostringstream ss;
    map < String, bool > colors;
    time_t startTimer, endTimer, candidateStart, candidateEnd, imageStart, imageEnd, targetStart, targetEnd;
    json::Array finalArray;
    
    //Choose to remove "green", "brown" and/or "gray" as defined in the HSV
    //value ranges seen in the inRange() function calls in RemoveColorsFromImage
    colors.insert( make_pair( "Green", true ) );
    colors.insert( make_pair( "Brown", true ) );
    colors.insert( make_pair( "Gray", true ) );
    
    //Start run timer
    time( &startTimer );
    
    //For every file in the directory specified in global variable *dir
    for ( long a = 0; a < NUM_FILES; a++ ) {
        //Generate image name string
        ss << a;
        string fullPath = String( dir ) + "/" + ss.str() + ".jpg";
        ss.str( "" );
        
        //Start image timer
        time( &imageStart );
        
        //Read an image
        Mat original = CreateMatFromImage( fullPath );
        
        /*
         Remove the desired colors from the image by scaling it
         into the HSV colorspace, creating a binary mask from
         the result, and overlaying it back onto the original.
         */
        Mat reducedColors = RemoveColorsFromImage( original, colors );
        Mat binary = BinaryImage( reducedColors );
        
        /*
         Store any candidate targets found in the image in an array
         Min and Max area values determined from size of target and
         height from which the image is taken (did not have time for
         this algorithm; supplied values were determined from test
         image generation suite inputs)
         */
        vector< Mat > candidates = FindCandidateTargets ( original, binary, MIN_AREA, MAX_AREA, WHITE, &posCounter, xPos, yPos );
        
        //Start candidate timer
        time( &candidateStart );
        
        //Write all the candidate targets to a directory
        for ( int i = 0; i < candidates.size(); i++ ) {
            ss << ( writeCandidateCounter + 1 );
            imwrite( candidateDir + ss.str() + ".jpg", candidates[i] );
            writeCandidateCounter++;
            ss.str( "" );
        }
        
        //stop candidate time
        time( &candidateEnd );
        double candidateTime = CalcTime( candidateStart, candidateEnd, "sec" );
        
        //Create new json array for each image
        json::Array targets;
        
        //For each candidate that has been designated as a target
        for ( int i = 0; i < candidates.size(); i++ ) {
            //start target timer
            time( &targetStart );
            
            String characterNames[CLUSTERS];
            int badFlag = 0;
            
            //Split the target into 5 different color bins and write each bin
            //to a separate Mat
            ss << readCandidateCounter+1;
            Mat can = imread( candidateDir + ss.str() + ".jpg" );
            vector< Mat > clusters = CreateClustersFromMat( can, CLUSTERS );
            readCandidateCounter++;
            ss.str( "" );
            
            //Attempt to determine a character name from any of the five bins
            for ( int x = 0; x < ( sizeof( characterNames )/sizeof( *characterNames ) ); x++ ) {
                if ( !clusters[x].empty() ) {
                    characterNames[x] = Identify( Mat(), clusters[x] );
                }
            }
            
            //Increment a counter if we found no character in a Mat
            for ( int x = 0; x < ( sizeof( characterNames )/sizeof( *characterNames ) ); x++ ) {
                if ( characterNames[x] == "BAD" ) {
                    badFlag++;
                }
            }
            
            //If they're all bad, toss out the target
            if ( badFlag == CLUSTERS ) {
                posCounter++;
                badFlag = 0;
                continue;
            }
            
            String conf[CLUSTERS];
            
            //Store the confidence rates in Strings
            for ( int x = 0; x < ( sizeof( characterNames )/sizeof( *characterNames ) ); x++ ) {
                if ( characterNames[x] != "BAD" )
                    conf[x] = characterNames[x].substr( 2, 2 );
                else
                    conf[x] = "00000";
            }
            
            //Assume conf1 is highest
            int topConf = 1;
            
            istringstream istrConfs[CLUSTERS];
            int confidence[CLUSTERS];
            
            //Take the String-based confidence rate and morph it into an Int
            //Store the Ints in another array
            for ( int x = 0; x < sizeof( istrConfs )/sizeof( *istrConfs ); x++ ) {
                istrConfs[x].str(conf[x]);
                istrConfs[x] >> confidence[x];
            }
            
            //Determine the highest confidence rate out of the five bins
            for ( int x = 0; x < sizeof( confidence )/sizeof( *confidence ); x++ ) {
                if ( x == 0 ) {
                    topConf = 0;
                    continue;
                }
                else if ( confidence[x] > confidence[topConf] ) {
                    topConf = x;
                }
            }
            
            //reset the bad character flag
            badFlag = 0;
            
            //if the OCR function lets the candidate pass, it's a target
            Mat targetCutout = candidates[i];
            
            Mat clearTarget = RemoveColorsFromImage( targetCutout, colors );
            Mat cleanTargetThresh = BinaryImage( clearTarget );
            
            //Determine shape of the target
            String shape = DetermineShape( cleanTargetThresh );
            
            //Determine target color and character (inner target) color
            String targetColor = DetectTargetColor( clearTarget, &bFinal, &gFinal, &rFinal );
            String characterColor = DetectCharacterColor( clearTarget, &bFinal, &gFinal, &rFinal );
            
            //Write the target out locally to a file for later evaluation if debugging enabled
            if ( verbose != 0 ) {
                if ( isdigit( fullPath.at( ( fullPath.length()-6 ) ) ) ) {
                    ss << finalCount+1;
                    imwrite( String( finalOut ) + "target" + ss.str() + characterNames[topConf] + "origin" + fullPath.substr( ( fullPath.length()-6 ),( fullPath.length()-6 ) ), targetCutout );
                    finalCount++;
                    ss.str( "" );
                }
                else {
                    ss << finalCount+1;
                    imwrite( String( finalOut ) + "target" + ss.str() + characterNames[topConf] + "origin" + fullPath.substr( ( fullPath.length()-5 ),( fullPath.length()-5 ) ), targetCutout );
                    finalCount++;
                    ss.str( "" );
                }
            }
            
            //stop target time
            time( &targetEnd );
            double targetTime = CalcTime( targetStart, targetEnd, "sec" );
            
            //save target data to JSON object
            ss << targetTime;
            json::Object myObject;
            string sub = characterNames[topConf].substr( 0,1 );
            myObject["letter"] = LowerLetter( sub );
            myObject["letter_color"] = characterColor;
            myObject["shape"] = shape;
            myObject["shape_color"] = targetColor;
            myObject["x"] = xPos[posCounter];
            myObject["y"] = yPos[posCounter];
            myObject["target_time"] = ss.str();
            posCounter++;
            ss.str( "" );
            
            //save target data to original image's JSON object
            targets.push_back( myObject );
        }
        
        //stop image time
        time( &imageEnd );
        double imageTime = CalcTime( imageStart, imageEnd, "sec" );
        
        //save calculation time data
        json::Object myObject;
        ss << candidateTime;
        myObject["candidate_time"] = ss.str();
        ss.str("");
        ss << imageTime;
        myObject["image_time"] = ss.str();
        ss.str("");
        myObject["targets"] = targets;
        
        finalArray.push_back( myObject );
    }
    
    //stop run time
    time( &endTimer );
    double runTime = CalcTime( startTimer, endTimer, "min" );
    printf( "Runtime: %.02lfmin\n", runTime );
    
    int closingFlag = 0;
    
    //output run time to JSON file and close it
    try {
        json::Object myObject;
        myObject["runtime"] = runTime;
        finalArray.push_back( myObject );
        String serialized_json = json::Serialize( finalArray );
        jsonOutput.open( jsonOut, ios::app );
        jsonOutput << serialized_json;
        jsonOutput.close();
    } catch(...) {
        printf("There was a problem with writing the runtime to the JSON file\n");
        printf("The runtime has been displayed to STDOUT\n");
        closingFlag = 1;
    }
    
    if (closingFlag == 1) {
        try {
            String serialized_json = json::Serialize( finalArray );
            jsonOutput.open( jsonOut, ios::app );
            jsonOutput << serialized_json;
            jsonOutput.close();
        } catch (...) {
            printf("Looks like there was a problem finishing up the JSON file too.. damn!");
            printf("No data for you..");
        }
    }
    
    return 0;
}