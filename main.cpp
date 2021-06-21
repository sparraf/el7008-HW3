/* Universidad de Chile - FCFM
 * EL7008 - Advanced Image Processing
 * Homework 3: Race classification
 *
 * Author: Sebastián Parra
 * 2018
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <bitset>
#include <iostream>
#include <opencv/ml.h>

// Function to create LBP image of a given source image. Note that
// border pixels are not processed and thus output img is of dimensions
// (src_image.rows - 1) x (src_image.cols - 1)
cv::Mat LBPTransform(const cv::Mat &src_image)
{
    cv::Mat src_gray;

    // Convert source to grayscale
    cv::cvtColor(src_image, src_gray, cv::COLOR_BGR2GRAY);

    cv::Mat LBP(src_gray.rows - 2, src_gray.cols - 2, CV_8UC1);

    // Iterate for all non-border pixels
    for (int r=1; r<src_gray.rows-1; r++)
    {
        for (int c=1; c<src_gray.cols-1; c++)
        {
            // Get center pixel and reset binary pattern
            float center = src_gray.at<float>(r,c);
            unsigned char binPatt = 0;

            // For each pixel, check its 8 neighbors in a clockwise manner
            int neighR = r-1;
            int neighC = c-1;
            for(int direction=0; direction<4; direction++)
            {
                for (int step=1; step<3; step++)
                {
                    switch (direction)
                    {
                        case 0:
                            neighC++;
                            break;
                        case 1:
                            neighR++;
                            break;
                        case 2:
                            neighC--;
                            break;
                        case 3:
                            neighR--;
                            break;
                    }
                    // Multiply binary pattern by 2 (equivalent to left bit shift)
                    binPatt *= 2;
                    // For each neighbor, check if I(neigh) > I(center). If true,
                    // append a 1 to the binary pattern
                    float neighbor = src_gray.at<float>(neighR, neighC);
                    if (neighbor > center)
                    {
                        binPatt++;
                    }
                }
            }

            // Insert LBP transform of center pixel in LBP imagen
            LBP.at<unsigned char>(r-1, c-1) = binPatt;
        }
    }
    return LBP;
}

// Function to create uniform LBP histogram of a given LBP image with a given uniform lookup table.
// Input image is divided in 4 regions, a uniform LBP histogram is computed for each one, and output is
// a concatenation of all 4 histograms.
cv::Mat uniformLBPHistogram(cv::Mat imLBP, cv::Mat lookupTable)
{
    cv::Mat uniformLBP, histogram;

    // Get uniform LBP values with lookup table and split matrix in 4 regions
    // (top-left, top-right, bottom-left, bottom-right)
    cv::Mat TL, TR, BL, BR;
    cv::LUT(imLBP, lookupTable, uniformLBP);

    TL = uniformLBP(cv::Rect(0, 0, uniformLBP.cols/2, uniformLBP.rows/2));
    TR = uniformLBP(cv::Rect(0, uniformLBP.cols/2, uniformLBP.cols/2, uniformLBP.rows/2));
    BL = uniformLBP(cv::Rect(uniformLBP.rows/2, 0, uniformLBP.cols/2, uniformLBP.rows/2));
    BR = uniformLBP(cv::Rect(uniformLBP.rows/2, uniformLBP.cols/2, uniformLBP.cols/2, uniformLBP.rows/2));

    // Create top-left, top-right, bottom-left and bottom-right histograms
    cv::Mat histTL, histTR, histBL, histBR;
    int histSize = 59;
    float range[] = {0, 59};
    const float* histRange = {range};

    cv::calcHist(&TL, 1, 0, cv::Mat(), histTL, 1, &histSize, &histRange, true, false);
    cv::calcHist(&TR, 1, 0, cv::Mat(), histTR, 1, &histSize, &histRange, true, false);
    cv::calcHist(&BL, 1, 0, cv::Mat(), histBL, 1, &histSize, &histRange, true, false);
    cv::calcHist(&BR, 1, 0, cv::Mat(), histBR, 1, &histSize, &histRange, true, false);

    // Concatenate histograms
    cv::hconcat(histTL.t(), histTR.t(), histogram);
    cv::hconcat(histogram, histBL.t(), histogram);
    cv::hconcat(histogram, histBR.t(), histogram);

    return histogram;
}

// Function to get a lookup table to map normal LBP values, to uniform LBP values, considering
// all non uniform values are mapped to 0
cv::Mat getUniformLBPTable()
{
    // Create lookup table for uniform LBP conversion
    cv::Mat lookup(1, 256, CV_8UC1);
    unsigned char uniformMap = 1; // Value to map next uniform value into
    for (int i=0; i<256; i++)
    {
        // Get a copy of i left-shifted and XOR it with original i value.
        auto original = (unsigned char) i;
        unsigned char shifted = original << 1;
        unsigned char transitions = original ^ shifted;
        size_t counts = std::bitset<8>(transitions).count();
        // If least significant bit is 1, substract a value to number of transitions
        if (i % 2 == 1)
        {
            counts--;
        }
        // If number is not uniform, map to 0. Else, map to current value set for next uniform bitstring
        if (counts > 2)
        {
            lookup.at<unsigned char>(0, i) = 0;
        }
        else
        {
            lookup.at<unsigned char>(0, i) = uniformMap;
            uniformMap++;
        }
    }
    return lookup;
}

int main() {

    // Make user choose to run binary or multiclass classification problem (1 is for binary, 2 for multiclass)
    int problemType;
    bool valid = false;
    std::cout << "Please select which classification problem to solve (1 or 2):" << std::endl;
    std::cout << "1. Binary classification problem (2 classes: Asian and Black)" << std::endl;
    std::cout << "2. Multiclass classification (5 classes: Asian, Black, Indian, White, and Others)" << std::endl;

    while (!valid) {
        if (!(std::cin >> problemType)) {
            std::cin.clear();
            std::cin.ignore();
            std::cout << "Please enter a valid number (1 or 2)" << std::endl;
        } else if (problemType > 2 || problemType < 1) {
            std::cin.clear();
            std::cin.ignore();
            std::cout << "Please enter a valid number (1 or 2)" << std::endl;
        } else {
            valid = true;
            std::cin.clear();
            std::cin.ignore();
        }
    }

    // Get LUT to transform LBP image into uniform LBP encoding
    cv::Mat lookupLBP = getUniformLBPTable();

    // Create features and labels matrices
    cv::Mat features, labels;
    cv::String path;
    std::string inputPath;

    if (problemType == 1)
        // CASE 1: Binary classification between "Asian" and "Black"
        std::cout << "Please write location of folder /Case1 (ex. ...myFolder/separated/Case1)" << std::endl;
    else if (problemType == 2)
        // CASE 2: Multiclass classification with all 5 labels
        std::cout << "Please write location of folder /separated (ex. ...myFolder/separated)" << std::endl;

    // Get images path
    std::getline(std::cin, inputPath);
    path = inputPath;

    // Get images
    std::vector<cv::String> fn;
    cv::glob(path, fn, true);
    int label = -1;
    for (size_t k = 0; k < fn.size(); ++k) {
        cv::Mat src = cv::imread(fn[k]);
        // Process only successful attempts
        if (src.empty()) {
            continue;
        }
        // Get LBP image
        cv::Mat LBP = LBPTransform(src);
        // Get uniform LBP histogram of LBP image
        cv::Mat hist = uniformLBPHistogram(LBP, lookupLBP);
        // Get feature value (0 for Asian, 1 for Black, 2 for Indian, 3 for Others, and 4 for White)
        if (k % 200 == 0) {
            label++;
        }
        cv::Mat currentLabel(1, 1, CV_32SC1, cv::Scalar(label));

        // Append to feature and label matrices
        if (k == 0) {
            features = hist;
            labels = currentLabel;
        } else {
            cv::vconcat(features, hist, features);
            cv::vconcat(labels, currentLabel, labels);
        }
    }

    // Create TrainData object. Observations are rows and features are columns
    cv::Ptr<cv::ml::TrainData> faceData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
    // Set 0.3 test ratio and shuffle data
    faceData->setTrainTestSplitRatio(0.7, true);

    // Convert labels to int. For some reason, they get treated as float if this is not done, which conflicts
    // with SVM, since SVM needs labels as int vector
    cv::Mat trainFeatures = faceData->getTrainSamples();
    cv::Mat trainLabels = faceData->getTrainResponses();
    trainLabels.convertTo(trainLabels, CV_32S);

    cv::Mat testFeatures = faceData->getTestSamples();
    cv::Mat testLabels = faceData->getTestResponses();
    testLabels.convertTo(testLabels, CV_32S);


    // Create SVM
    auto svm = cv::ml::SVM::create();
    // SVM parameters
    svm->setType(svm->C_SVC); // C-Support Vector Classifier
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100000, 1e-6)); // max iterations, tolerance
    svm->setKernel(svm->LINEAR);

    float binaryCValues[] = {1e-6, 1e-5, 1e-4, 0.001, 0.01};
    float multiclassCValues[] = {1e-8, 1e-7, 1e-6, 1e-5, 1e-4};
    float CValues[5];

    if (problemType == 1)
        std::copy(binaryCValues, binaryCValues+5, CValues);
    else if (problemType == 2)
        std::copy(multiclassCValues, multiclassCValues+5, CValues);

    // Train on different parameter configurations
    for (const auto& C : CValues) {
        // Set C value
        svm->setC(C);

        svm->train(trainFeatures, 0, trainLabels);

        // Get train and test errors
        float trainErr = svm->calcError(faceData, false, cv::noArray());
        float testErr = svm->calcError(faceData, true, cv::noArray());

        // Show training and validation error
        std::cout << "Accuracy (%) for linear SVM classifier with parameters" << std::endl;
        std::cout << "C = " << C << std::endl;
        std::cout << "Train: " << 100-trainErr << ", Test: " << 100-testErr << std::endl;
        std::cout << std::endl;
    }

    // Create RF classifier
    auto rf = cv::ml::RTrees::create();

    rf->train(trainFeatures, 0, trainLabels);

    float trainErr = rf->calcError(faceData, false, cv::noArray());
    float testErr = rf->calcError(faceData, true, cv::noArray());

    std::cout << "Accuracy (%) for RF classifier with default parameters" << std::endl;
    std::cout << "Train: " << 100-trainErr << ", Test: " << 100-testErr << std::endl;
    return 0;
}

