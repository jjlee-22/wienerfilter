/**
* @summary This is an implementation of the Wiener filter 
* @author Jonathan Lee
* @credits https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
* https://docs.opencv.org/3.4/de/d3c/tutorial_out_of_focus_deblur_filter.html
*/
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
using namespace cv;

static void on_radius_trackbar(int radius_slider, void* imgOrig);
static void on_snr_trackbar(int snr_slider, void* image);
void displayFiltered(Mat& imgMod);
void filter(const Mat& inputImg, Mat& outputImg, const Mat& H);
void getWienerFilter(Size bbox, Mat& imgOut);

int radius_slider = 64; // Default radius slider value
int snr_slider = 1200; // Default signal noise ratio value

// main - implement the Wiener filter
// preconditions: input image
// postconditions: output image filtered by Wiener function
int main(int argc, char *argv[])
{
    Mat imgOrig = imread("original.jpg", IMREAD_GRAYSCALE); // input of the image file must be named as original.jpg

    //Trackbar creation
    namedWindow("Wiener Filter");
    char TrackbarNameRadius[50];
    char TrackbarNameSNR[50];
    sprintf_s(TrackbarNameRadius, "Radius");
    sprintf_s(TrackbarNameSNR, "SNR");
    createTrackbar(TrackbarNameRadius, "Wiener Filter", &radius_slider, 130, on_radius_trackbar, &imgOrig);
    createTrackbar(TrackbarNameSNR, "Wiener Filter", &snr_slider, 2000, on_snr_trackbar, &imgOrig);

    on_radius_trackbar(radius_slider, &imgOrig);
    on_snr_trackbar(snr_slider, &imgOrig);
    waitKey(0);

    return 0;
}

// on_radius_trackbar - filter the image every time the radius value is changed
// preconditions: radius slider, input image
// postconditions: results of the deblurred image
static void on_radius_trackbar(int radius_slider, void* image)
{
    Mat imgOrig = *(Mat*)image;
    Mat imgMod;
    Mat imgFiltered; 
    Rect bbox = Rect(0, 0, imgOrig.cols, imgOrig.rows);

    // Calculate the wiener filter
    getWienerFilter(bbox.size(), imgFiltered);
    filter(imgOrig(bbox), imgMod, imgFiltered); // filter the image using the wiener function
    displayFiltered(imgMod);
}

// on_snr_trackbar- filter the image every time the signal noise ration value is changed
// preconditions: snr slider, input image
// postconditions: results of the deblurred image
static void on_snr_trackbar(int snr_slider, void* image)
{
    Mat imgOrig = *(Mat*)image;
    Mat imgMod;
    Mat imgFiltered;
    Rect bbox = Rect(0, 0, imgOrig.cols, imgOrig.rows);

    // Calculate the Wiener filter
    getWienerFilter(bbox.size(), imgFiltered);
    filter(imgOrig(bbox), imgMod, imgFiltered); // filter the image using the wiener function
    displayFiltered(imgMod);
}

// filter - filters the blurred image using the wiener function Hw through the frequency domain 
// CREDITS TO VLAD KARPUSHIN for this code snippet. I had a hard time implementing this part on my own
// precondtions: blurred image, image to be restored, wiener function
// postconditions: deblurred image using Wiener filter
void filter(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    cv::dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);

    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}

// getWienerFilter - get the PSF and then calculate the Wiener filter which is a fourier's transform of the degradation model
// preconditions: region of interest, output image, signal noise ratio (power spectrum of noise divided by the signal
// postconditions: calculate the wiener filter
void getWienerFilter(Size bbox,  Mat& imgOut)
{
    // Calculate the PSF degradation model
    Mat freqPSF;
    Mat cI;
    int radW = bbox.width;
    int radH = bbox.height;
    Point point(radW/2, radH/2); //create new point with radius of the roi/2 the size
    Mat psf(bbox, CV_32F, Scalar(0)); // create new image
    circle(psf, point, radius_slider, 255, -1, 5); //create a circle with img h, center of circle denoted by point, radius of circle, black color, thickness of 5
    psf = psf / sum(psf); //divide h by the sum of all array element in the image pixel

    // intermodulate diagonal regions in the spatial domain for PSF (from the OpenCV fourier tutorial doc)
    freqPSF = psf.clone();

    int cx = freqPSF.cols / 2;
    int cy = freqPSF.rows / 2;
    Mat q0(freqPSF, Rect(0, 0, cx, cy)); // shift first quadrant
    Mat q1(freqPSF, Rect(cx, 0, cx, cy)); // shift second quadrant
    Mat q2(freqPSF, Rect(0, cy, cx, cy)); // shift third quadrant
    Mat q3(freqPSF, Rect(cx, cy, cx, cy)); // shift fourth quadrant

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    Mat planes[2] = { freqPSF.clone(), Mat::zeros(freqPSF.size(), CV_32F) };
    cv::merge(planes, 2, cI);
    cv::dft(cI, cI);
    cv::split(cI, planes);

    // Wiener filter = H / (H^2 + 1/SNR)
    Mat degrad;
    cv::pow(cv::abs(planes[0]), 2, degrad);
    degrad += 1.0 / double(snr_slider);
    cv::divide(planes[0], degrad, imgOut);
}

// displayFiltered - display the output image
// preconditions: modified (filtered) image
// postconditions: displays and saves the output image file
void displayFiltered(Mat& imgMod)
{
    imgMod.convertTo(imgMod, CV_8U);
    resize(imgMod, imgMod, Size(), 0.3, 0.3);
    imshow("Wiener Filter", imgMod);
    imwrite("filtered.jpg", imgMod);
}
