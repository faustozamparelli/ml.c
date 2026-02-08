#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {
	{0, 0},
	{1, 2},
	{2, 4},
	{3, 6},
	{4, 8},
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float() {
	return (float)rand()/(float)RAND_MAX;
}

float loss(float w, float b) {
	float result = 0.0f;

	for (size_t i = 0; i < train_count; i++) {
		float x = train[i][0];
		float y = x*w+b;
		float d = y - train[i][1];
		result += d*d;
	}
	result /= train_count;
	return result;
}


int main() {
	// y = x*w;
	// srand(time(0));
	// rand();
	srand(200);
	float w = rand_float()*10.0f;
	//float w = 100;
	float b = rand_float()*5.0f;
	float eps = 1e-5;
	float rate = 1e-2;

	for (size_t i=0; i<1000; i++) {
		float l = loss(w,b); 
		float dw_loss = (loss(w + eps, b) - l)/eps;
		float db_loss = (loss(w, b + eps) - l)/eps;
		w -= rate*dw_loss;
		b -= rate*db_loss;
		printf("loss%zu: %f, weight: %f, bias: %f\n",i, loss(w, b), w, b);
	}
	printf("------------\n%f\n", w);
	
  return 0;
}
