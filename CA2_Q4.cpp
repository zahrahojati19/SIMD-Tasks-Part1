#include "iostream"
#include "stdio.h"
#include <sys/time.h>
#include <x86intrin.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void merge_Parallel2(Mat img01, Mat img02) {
    struct timeval start, end;

    __m128 *pSrc01, *pSrc02;
	__m128 *pRes;
	__m128 m1, m2, m3;
    pSrc01 = ( __m128 *) img01.data;
    pSrc02 = ( __m128 *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8UC3);
    __m128 alpha;
    alpha = _mm_set1_ps(0.25);
    gettimeofday(&start, NULL);
	for (int i = 0; i < img02.rows; i++)
		for (int j = 0; j < img02.cols/16*3; j++) {
			m1 = _mm_load_ps((const float*)(pSrc01 + i*img01.cols/16*3 + j));
            m2 = _mm_load_ps((const float*)(pSrc02 + i*img02.cols/16*3 + j));
            //cout << pSrc02 + i*img02.cols/16*3 + j << " " << pSrc01 + i*img01.cols/16*3 + j << endl;
           // m2 = _mm_mul_ps(m2, alpha);
			//m3 = _mm_add_ps(m1, m2);
			_mm_store_ps((float*)(pSrc01 + i*img01.cols/16*3 + j), m2);
		}
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = (unsigned char *)pSrc01;
    namedWindow("imageParallel_Q4", WINDOW_AUTOSIZE);
    imshow("imageParallel_Q4", out_img);
}

void merge_Parallel_Gray(Mat img01, Mat img022) {
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
    float sum =0;
    __m128 alpha = _mm_set1_ps(0.25);
    gettimeofday(&start, NULL);
	for (int i = 0; i < img022.rows; i++)
		for (int j = 0; j < img022.cols/16; j++) {
			m1 = _mm_loadu_si128(pSrc01 + i*img01.cols/16 + j);
            m2 = _mm_loadu_si128(pSrc02 + i*img02.cols/16 + j);
            m2 = _mm_mul_epu32(m2, _mm_castps_si128(alpha));
			m3 = _mm_adds_epu8(m1, m2);
			_mm_storeu_si128 (pSrc01 + i*img01.cols/16 + j, m3);
		}
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = (unsigned char *)pSrc01;
    namedWindow("imageParallel2_Q4", WINDOW_AUTOSIZE);
    imshow("imageParallel2_Q4", out_img);
}

void merge_Parallel(Mat img01, Mat img022) {
    struct timeval start, end;

    Mat img02(Size(img01.cols,img01.rows),img022.type());
    img02 = Scalar(0);
    img022.copyTo(img02(Rect(0,0,img022.cols,img022.rows)));

    __m128i *pSrc01, *pSrc02;
	__m128i *pRes;
	__m128i m1, m2, m3;
    pSrc01 = ( __m128i *) img01.data;
    pSrc02 = ( __m128i *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8UC3);
    float sum =0;
    __m128 alpha = _mm_set1_ps(0.25);
    gettimeofday(&start, NULL);
	for (int i = 0; i < img022.rows; i++)
		for (int j = 0; j < img022.cols/16*3; j++) {
			m1 = _mm_loadu_si128(pSrc01 + i*img01.cols/16*3 + j);
            m2 = _mm_loadu_si128(pSrc02 + i*img02.cols/16*3 + j);
            m2 = _mm_mul_epu32(m2, _mm_castps_si128(alpha));
			m3 = _mm_adds_epu8(m1, m2);
			_mm_storeu_si128 (pSrc01 + i*img01.cols/16*3 + j, m3);
		}
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = (unsigned char *)pSrc01;
    namedWindow("imageParallel2_Q4", WINDOW_AUTOSIZE);
    imshow("imageParallel2_Q4", out_img);
}

void merge_Serial_Gray(Mat img01, Mat img02) {
    struct timeval start, end;

    unsigned char *in_image01;
    unsigned char *in_image02;
    in_image01 = (unsigned char *) img01.data;
    in_image02 = (unsigned char *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8U);

    gettimeofday(&start, NULL);
    for(int i = 0; i < img02.rows; i++)
        for(int j = 0; j < img02.cols; j++) 
            in_image01[i * img01.cols + j] = in_image01[i * img01.cols + j] + in_image02[i * img02.cols + j]*0.25;
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = in_image01;
    namedWindow("imageSerial_Q4", WINDOW_AUTOSIZE);
    imshow("imageSerial_Q4", out_img);
}

void merge_Serial(Mat img01, Mat img02) {
    struct timeval start, end;

    unsigned char *in_image01;
    unsigned char *in_image02;
    in_image01 = (unsigned char *) img01.data;
    in_image02 = (unsigned char *) img02.data;
    Mat out_img(img01.rows, img01.cols, CV_8UC3);

    gettimeofday(&start, NULL);
    for(int i = 0; i < img02.rows; i++)
        for(int j = 0; j < img02.cols*3; j++) 
            in_image01[(i*3) * img01.cols + j] = in_image01[i*3 * img01.cols + j] + in_image02[i*3 * img02.cols + j]*0.25;
    gettimeofday(&end, NULL);

    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);

    out_img.data = in_image01;
    namedWindow("imageSerial_Q4", WINDOW_AUTOSIZE);
    imshow("imageSerial_Q4", out_img);
}

void print_name() {
    cout << "Zahra Hojati\nStudent ID : 810199403\n" << endl;
}

void print_delimiter() {
    cout << "***********************" << endl;
}

int main() {
    Mat img01 = imread("Image_01.png", IMREAD_GRAYSCALE);
    Mat img02 = imread("Image_02.png", IMREAD_GRAYSCALE);
    Mat img011 = imread("Image_01.png", IMREAD_GRAYSCALE);
    Mat img022 = imread("Image_02.png", IMREAD_GRAYSCALE);

    print_name();
    print_delimiter();
    //merge_Parallel(img011, img022);
    merge_Parallel_Gray(img011, img022);
    print_delimiter();
    //merge_Serial(img01, img02);
    merge_Serial_Gray(img01, img02);
    waitKey(0);
    return 0;
}