#define NN_IMPLEMENTATION
#include "nn.h"

typedef struct {
	Mat a0;
	Mat w1, b1, a1;
	Mat w2, b2, a2;
}Xor;

//create the matrix for each layer, weight and bias
Xor xor_alloc() {
	Xor m;
	m.a0 = mat_alloc(1,2);
	m.w1 = mat_randalloc(2, 2, -1, 1);
	m.b1 = mat_randalloc(1, 2, -1, 1);

	m.a1 = mat_alloc(1,2);
	m.w2 = mat_randalloc(2, 1, -1, 1);
	m.b2 = mat_randalloc(1, 1, -1, 1);

  m.a2 = mat_alloc(1,1);
	return m;
}

// compute the output from the input of the NN
void forward_xor(Xor m) {
	mat_dot(m.a1,m.a0,m.w1);
	mat_sum(m.a1, m.b1);
	mat_sig(m.a1);
	mat_dot(m.a2,m.a1,m.w2);
	mat_sum(m.a2, m.b2);
	mat_sig(m.a2);
}

// copute the cost based on the input ti and the target output to
float cost(Xor m, Mat ti, Mat to) {
	NN_ASSERT(ti.rows == to.rows);

	float c = 0;
	for (size_t i=0; i<ti.rows;i++) {
		Mat x = mat_row(ti,i);
		Mat y = mat_row(to,i);

		mat_copy(m.a0, x);
		forward_xor(m);

		for (size_t j=0; j<to.cols; j++) {
			float d = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
			c+= d*d;
		}
	}
	return c/ti.rows;
}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to) {
	float c = cost(m, ti, to);
	float saved;

	for (size_t i=0; i<m.w1.rows; i++) {
		for (size_t j=0; j<m.w1.cols; j++) {
			saved = MAT_AT(m.w1, i, j);
			MAT_AT(m.w1, i, j) += eps;
			MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.w1, i, j) = saved;
	  }
	}
  for (size_t i=0; i<m.b1.rows; i++) {
		for (size_t j=0; j<m.b1.cols; j++) {
			saved = MAT_AT(m.b1, i, j);
			MAT_AT(m.b1, i, j) += eps;
			MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.b1, i, j) = saved;
	  }
	}
	for (size_t i=0; i<m.w2.rows; i++) {
		for (size_t j=0; j<m.w2.cols; j++) {
			saved = MAT_AT(m.w2, i, j);
			MAT_AT(m.w2, i, j) += eps;
			MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.w2, i, j) = saved;
	  }
	}
  for (size_t i=0; i<m.b2.rows; i++) {
		for (size_t j=0; j<m.b2.cols; j++) {
			saved = MAT_AT(m.b2, i, j);
			MAT_AT(m.b2, i, j) += eps;
			MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.b2, i, j) = saved;
	  }
	}
}

void train(Xor m, Xor g, float rate) {
	for (size_t i=0; i<m.w1.rows; i++) {
		for (size_t j=0; j<m.w1.cols; j++) {
			MAT_AT(m.w1, i, j) -= MAT_AT(g.w1, i, j) * rate;
	  }
	}
  for (size_t i=0; i<m.b1.rows; i++) {
		for (size_t j=0; j<m.b1.cols; j++) {
			MAT_AT(m.b1, i, j) -= MAT_AT(g.b1, i, j) * rate;
	  }
	}
	for (size_t i=0; i<m.w2.rows; i++) {
		for (size_t j=0; j<m.w2.cols; j++) {
			MAT_AT(m.w2, i, j) -= MAT_AT(g.w2, i, j) * rate;
	  }
	}
  for (size_t i=0; i<m.b2.rows; i++) {
		for (size_t j=0; j<m.b2.cols; j++) {
			MAT_AT(m.b2, i, j) -= MAT_AT(g.b2, i, j) * rate;
	  }
	}
}


float td[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0,
};

int main(void) {
	Mat ti = {
		.rows = 4,
		.cols = 2,
		.stride = 3,
		.elements = td,
	};
	Mat to = {
		.rows = 4,
		.cols = 1,
		.stride = 3,
		.elements = td+2,
	};

	Xor m = xor_alloc();
	Xor g = xor_alloc();
	for (size_t i=0; i<10000; i++) {
		finite_diff(m, g, 1e-3, ti, to);
		train(m, g, 1e-1);
	}
	for (size_t i = 0; i < ti.rows; i++) {
			Mat x = mat_row(ti, i);
			mat_copy(m.a0, x);
			forward_xor(m);
			printf("%f ^ %f = %f\n", MAT_AT(ti, i, 0), MAT_AT(ti, i, 1), MAT_AT(m.a2, 0, 0));
	}
	return 0;
}
