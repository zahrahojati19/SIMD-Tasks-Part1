#include "iostream"
#include "stdio.h"
#include <sys/time.h>
#include <vector>
#include <x86intrin.h>
#include <cmath>
#include <pmmintrin.h>

using namespace std;

typedef union {
    __m128 sp128;
    float m128_sp32[4];   
} floatVec;

vector<floatVec> initiat_array() {
    vector <floatVec> vec (2 << 20);
    float LO = -(100 + 1.18 / 1e38);
	float HI = 100 + 1.18 / 1e38;
	for (int i = 0; i < vec.size(); i++) {
		vec[i].m128_sp32[0] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		vec[i].m128_sp32[1] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		vec[i].m128_sp32[2] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
		vec[i].m128_sp32[3] = LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
	}
    return vec;
}

void serial_exe(vector<floatVec> vec) {
    struct timeval start, end;
    int index = 0;
    float min = vec[0].m128_sp32[0];
    gettimeofday(&start, NULL);
    for (int i = 0; i < vec.size(); i++) {
        for(int j = 0; j < 4; j++) {
            if(vec[i].m128_sp32[j] <= min) {
                min = vec[i].m128_sp32[j];
                index = i;
            }
        }
    }
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Minimum Value of the array: %f, at index: %d\n", min, index);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);
}

void parallel_exe(vector<floatVec> vec) {
    struct timeval start, end;
    int index;
    __m128 min = vec[0].sp128;
    __m128 temp = min;
    float flag;
    gettimeofday(&start, NULL);
    for(int i = 0; i < vec.size(); i++) {
        /*flag = _mm_cmpgt_ps (min, vec[i].sp128);
        if(flag[0]!=0) min[0] = vec[i].sp128[0];
        if(flag[1]!=0) min[1] = vec[i].sp128[1];
        if(flag[2]!=0) min[2] = vec[i].sp128[2];
        if(flag[3]!=0) min[3] = vec[i].sp128[3];*/
        min = _mm_min_ps (min, vec[i].sp128);
        //if(_mm_ucomineq_ss (min, temp)) index = i;
        if(temp[2] != min[2]) index = i;
        temp = min;
    }
    flag = min[0];
    for(int i = 1; i < 4; i++)
        if(flag > min[i]) flag = min[i];
    
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Minimum Value of the array: %f, at index: %d\n", flag, index);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);
}

float calculate_serial_avg(vector<floatVec> vec) {
    struct timeval start, end;
    float sum = 0, avg;
    gettimeofday(&start, NULL);
    for(int i = 0; i < vec.size(); i++)
        for(int j = 0; j < 4; j++)
            sum += vec[i].m128_sp32[j];
    avg = sum / (vec.size()*4);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Serial Average Value of the array: %f\n", avg);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);

    return avg;
}

float calculate_parallel_avg(vector<floatVec> vec) {
    struct timeval start, end;
    __m128 sum = vec[0].sp128;
    float avg = 0;
    gettimeofday(&start, NULL);
    for (int i = 1; i < vec.size(); i++)
        sum = _mm_add_ps (sum, vec[i].sp128);
    for(int i = 0; i < 4; i++)
        avg += sum[i];
    //sum = _mm_hadd_ps(sum, sum);
    //sum = _mm_hadd_ps(sum, sum);
    //avg = _mm_cvtss_f32(sum);
    avg = avg / (vec.size()*4);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Parallel Average Value of the array: %f\n", avg);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);

    return avg;
}

void calculate_parallel_dev(vector<floatVec> vec, float avg) {
    struct timeval start, end;
    __m128 sum, tmp;
    __m128 temp, divisor;
    float dev = 0;
    temp = _mm_set1_ps(avg);
    divisor = _mm_set1_ps(float((vec.size()*4)));
    sum = _mm_setzero_ps();
    gettimeofday(&start, NULL);
    for(int i = 0; i < vec.size(); i++) {
        tmp = _mm_sub_ps (vec[i].sp128, temp);
        tmp = _mm_mul_ps (tmp, tmp);
        sum = _mm_add_ps (sum, tmp);
    }
    sum = _mm_div_ps (sum, divisor);
    //sum = _mm_sqrt_ps (sum);
    for(int i = 0; i < 4; i++)
        dev += sum[i];
    dev = sqrt(dev);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Parallel Standard Devation Value of the array: %f\n", dev);
    printf("Parallel Execution time is %ld s and %ld micros\n\n", seconds, micros);
}

void calculate_serial_dev(vector<floatVec> vec, float avg) {
    struct timeval start, end;
    float dev, sum = 0;

    gettimeofday(&start, NULL);
    for(int i = 0; i < vec.size(); i++) 
        for(int j = 0; j < 4; j++) 
            sum += pow((vec[i].m128_sp32[j] - avg),2);
    sum = sum / (vec.size()*4);
    dev = sqrt(sum);
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    printf("Serial Standard Devation Value of the array: %f\n", dev);
    printf("Serial Execution time is %ld s and %ld micros\n\n", seconds, micros);
}

void print_name() {
    cout << "Zahra Hojati\nStudent ID : 810199403\n" << endl;
}

void print_delimiter() {
    cout << "***********************" << endl;
}

int main() {
    vector<floatVec> vec;
    float avg;

    print_name();
    print_delimiter();
    vec = initiat_array();
    parallel_exe(vec);
    print_delimiter();
    serial_exe(vec);
    print_delimiter();
    avg = calculate_parallel_avg(vec);
    calculate_parallel_dev(vec, avg);
    print_delimiter();
    avg = calculate_serial_avg(vec);
    calculate_serial_dev(vec, avg);
    print_delimiter();
}