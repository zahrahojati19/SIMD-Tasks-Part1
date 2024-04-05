#include "iostream"
#include "stdio.h"
#include <sys/time.h>
#include <x86intrin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void sub_Parallel_Gray(Mat img01, Mat img022) {
    struct timeval start, end;

    Mat img02(Size(img01.cols,img01.rows),img022.type());
    img02 = Scalar(0);
    img022.copyTo(img02(Rect(0,0,img022.cols,img022.rows)));

    __m128i *pSrc01, *pSrc02;
	__m128i *pRes;
	__m128i m1, m2, m3;
    pSrc01 = ( __m128i *) img01.data;
    pSrc02 = ( __m128i *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8U);

    gettimeofday(&start, NULL);
	for (int i = 0; i < img022.rows; i++)
		for (int j = 0; j < img022.cols/16; j++) {
			m1 = _mm_loadu_si128(pSrc01 + i*img01.cols/16 + j);
            m2 = _mm_loadu_si128(pSrc02 + i*img02.cols/16 + j);
			m3 = _mm_sub_epi8(m1, m2);
			_mm_storeu_si128 (pSrc01 + i*img01.cols/16 + j, m3);
		}
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = (unsigned char *)pSrc01;
    namedWindow("imageParallel_Q3", WINDOW_AUTOSIZE);
    imshow("imageParallel_Q3", out_img);
}

void sub_Parallel(Mat img01, Mat img022) {
    struct timeval start, end;

    Mat img02(Size(img01.cols,img01.rows),img022.type());
    img02 = Scalar(0);
    img022.copyTo(img02(Rect(0,0,img022.cols,img022.rows)));

    __m128i *pSrc01, *pSrc02, *out;
	__m128i *pRes;
	__m128i m1, m2, m3;
    pSrc01 = ( __m128i *) img01.data;
    pSrc02 = ( __m128i *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8UC3);
    out = ( __m128i *) out_img.data;

    gettimeofday(&start, NULL);
	for (int i = 0; i < img022.rows; i++)
		for (int j = 0; j < img022.cols/16*3; j++) {
			m1 = _mm_loadu_si128(pSrc01 + i*img01.cols/16*3 + j);
            m2 = _mm_loadu_si128(pSrc02 + i*img02.cols/16*3 + j);
			m3 = _mm_subs_epu8(m1, m2);
            for(int l=0;l<16;l++)
                m3[l] = abs(m3[l]);
			_mm_storeu_si128 (pSrc01 + i*img01.cols/16*3 + j, m3);
		}
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = (unsigned char *)pSrc01;
    namedWindow("imageParallel_Q3", WINDOW_AUTOSIZE);
    imshow("imageParallel_Q3", out_img);
}

void sub_Serial_Gray(Mat img01, Mat img02) {
    struct timeval start, end;

    unsigned char *in_image01;
    unsigned char *in_image02;
    in_image01 = (unsigned char *) img01.data;
    in_image02 = (unsigned char *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8U);

    gettimeofday(&start, NULL);
    for(int i = 0; i < img02.rows; i++)
        for(int j = 0; j < img02.cols; j++) 
            in_image01[i * img01.cols + j] = abs(in_image01[i * img01.cols + j] - in_image02[i * img02.cols + j]);
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = in_image01;
    namedWindow("imageSerial_Q3", WINDOW_AUTOSIZE);
    imshow("imageSerial_Q3", out_img);
}

void sub_Serial(Mat img01, Mat img02) {
    struct timeval start, end;

    unsigned char *in_image01;
    unsigned char *in_image02;
    unsigned char *out;
    in_image01 = (unsigned char *) img01.data;
    in_image02 = (unsigned char *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8UC3);
    out = (unsigned char *)out_img.data;

    gettimeofday(&start, NULL);
    for(int i = 0; i < img02.rows; i++){
        for(int j = 0; j < img02.cols*3; j++) 
            in_image01[(i*3) * img01.cols + j] = (abs(in_image01[i*3 * img01.cols + j] - in_image02[i*3 * img02.cols + j]));
        //for(int j = 0; j < img02.cols/16*3; j++) 
          //  for(int l=0;l<16;l++)
           // in_image01[(i*3) * img01.cols/16+16%l + j] = (abs(in_image01[i*3 * img01.cols/16+16%l + j] - in_image02[i*3 * img02.cols/16+16%l + j]));
        ;}
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = in_image01;
    namedWindow("imageSerial_Q3", WINDOW_AUTOSIZE);
    imshow("imageSerial_Q3", out_img);
}

void print_name() {
    cout << "Zahra Hojati\nStudent ID : 810199403\n" << endl;
}

void print_delimiter() {
    cout << "***********************" << endl;
}

int main() {
    Mat img01 = imread("Image_01.png");//, IMREAD_GRAYSCALE);
    Mat img02 = imread("Image_02.png");//, IMREAD_GRAYSCALE);
    Mat img011 = imread("Image_01.png");//, IMREAD_GRAYSCALE);
    Mat img022 = imread("Image_02.png");//, IMREAD_GRAYSCALE);

    print_name();
    print_delimiter();
    sub_Parallel(img01, img02);
    //sub_Parallel_Gray(img01, img02);
    print_delimiter();
    sub_Serial(img011, img022);
    //sub_Serial_Gray(img011, img022);
    waitKey(0);
    return 0;
}