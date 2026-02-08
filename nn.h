#ifndef NN_H
#define NN_H
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif //NN_MALLOC
#ifndef NN_ASSERT 
#define NN_ASSERT assert
#endif //NN_ASSERT

#define MAT_AT(m, i, j) (m).elements[(i)*m.stride + (j)]
#define MAT_PRINT(m) mat_print(m, #m)


typedef struct{
	size_t rows;
	size_t cols;
	size_t stride;
	float *elements;
} Mat;

float rand_float();
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);  
void mat_rand(Mat m, float low, float high);
Mat mat_randalloc(size_t rows, size_t cols, float low, float high);
void mat_fill(Mat m, float x);
void mat_print(Mat m, const char *name);
void mat_dot(Mat res, Mat a, Mat b);
void mat_sum(Mat res, Mat b);
void mat_sig(Mat res);
Mat mat_row(Mat m, size_t row);
Mat mat_col(Mat m, size_t col);
void mat_copy(Mat res, Mat m);

#endif //NN_H

#define NN_IMPLEMENTATION
#ifdef NN_IMPLEMENTATION
float rand_float() {
	return (float)rand()/(float)RAND_MAX;
}
float sigmoidf(float x) {
	return 1.0f / (1.0f + expf(-x));
}
Mat mat_alloc(size_t rows, size_t cols) {
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.stride = m.cols;
	m.elements = NN_MALLOC(sizeof(*m.elements)*rows*cols);
	NN_ASSERT(m.elements != NULL);
	return m;
}
void mat_rand(Mat m, float low, float high){
	for (size_t i=0; i<m.rows; i++) {
		for (size_t j=0; j<m.cols; j++) {
			MAT_AT(m, i, j) = rand_float()*(high-low)+low;
		}
	}
}
Mat mat_randalloc(size_t rows, size_t cols, float low, float high) {
	Mat m = mat_alloc(rows, cols);
	mat_rand(m, low, high);
	return m;
}
void mat_print(Mat m, const char *name){
	printf("%s: [\n", name);
	for (size_t i=0; i<m.rows; i++) {
		for (size_t j=0; j<m.cols; j++) {
			printf("    %f", MAT_AT(m, i, j));
		}
		printf("\n");
	}
	printf("]\n");
}
void mat_fill(Mat m, float x) {
	for (size_t i=0; i<m.rows; i++) {
		for (size_t j=0; j<m.cols; j++) {
			 MAT_AT(m, i, j) = x;
		}
	}
}
void mat_dot(Mat res, Mat a, Mat b){
	NN_ASSERT(a.cols == b.rows);
	NN_ASSERT(res.rows == a.rows);
	NN_ASSERT(res.cols == b.cols);
	for (size_t i=0; i<res.rows; i++) {
		for (size_t j=0; j<res.cols; j++) {
			MAT_AT(res, i, j) = 0;
			for (size_t k=0; k<a.cols; k++) {
				MAT_AT(res, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j) ;
			}
		}
	}
}
void mat_sum(Mat res, Mat b){
	NN_ASSERT(res.rows == b.rows);
	NN_ASSERT(res.cols == b.cols);
	for (size_t i=0; i<res.rows; i++) {
		for (size_t j=0; j<res.cols; j++) {
			 MAT_AT(res,i,j) += MAT_AT(b,i,j);
		}
	}
}
void mat_sig(Mat res){
	for (size_t i=0; i<res.rows; i++) {
		for (size_t j=0; j<res.cols; j++) {
        MAT_AT(res, i, j) = sigmoidf(MAT_AT(res, i, j));
		}
	}
}
Mat mat_row(Mat m, size_t row) {
	return (Mat) {
		.rows = 1,
		.cols = m.cols,
		.stride = m.cols,
		.elements = &MAT_AT(m, row, 0),
	};
}
void mat_copy(Mat res, Mat m) {
	NN_ASSERT(res.rows == m.rows);
	NN_ASSERT(res.cols == m.cols);
	for (size_t i=0; i<res.rows; i++) {
		for (size_t j=0; j<res.cols; j++) {
			MAT_AT(res, i, j) = MAT_AT(m, i, j);
		}
	}
};
#endif
