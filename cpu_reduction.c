
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SIZE 10000

double get_clock() {
  struct timeval tv; int ok;
  ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { printf("gettimeofday error"); }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int cpureduction(int *data, int length, char op) {
    int stride, i;

    for (stride = 1; stride < length; stride *= 2) {
        for (i = 0; i + stride < length; i += 2 * stride) {
            switch (op) {
                case '+':  // Sum
                    data[i] = data[i] + data[i + stride];
                    break;
                case '*':  // Product
                    data[i] = data[i] * data[i + stride];
                    break;
                case 'm':  // Min
                    data[i] = (data[i] < data[i + stride]) ? data[i] : data[i + stride];
                    break;
                case 'M':  // Max
                    data[i] = (data[i] > data[i + stride]) ? data[i] : data[i + stride];
                    break;
                default:
                    printf("Unsupported operation.\n");
                    exit(1);
            }
        }
    }

    return data[0];
}

int main() {
  int *data=malloc(sizeof(int)*SIZE);
    char op;

    for (int i = 0; i < SIZE; i++) {
        data[i] = 1;
	printf(" %d",data[i]);
    }
    printf("\n");
    printf("Enter the reduction operation (+, *, m for min, M for max): ");
    scanf(" %c", &op);

    double t0= get_clock();
    int result = cpureduction(data, SIZE, op);
    double t1=get_clock();
    
    printf("Reduction result: %d\n", result);
    printf("Time: %f ns\n", 1000000000.0*(t1 - t0));
    free(data);
    return 0;
}
