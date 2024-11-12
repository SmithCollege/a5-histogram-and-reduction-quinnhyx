#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define SIZE 1000000
#define RANGE 10

// Get time function
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, NULL);
    if (ok < 0) { 
        printf("gettimeofday error\n");
        exit(1); // exit on error
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// Compute histogram
void computeHistogram(int *data, int *histogram, int size, int range) {
    // Initialize histogram with zeros
    for (int i = 0; i < range; i++) {
        histogram[i] = 0;
    }

    // Count the frequency of each value in the data
    for (int i = 0; i < size; i++) {
        histogram[data[i]]++;
    }
}

int main() {
    // Allocate memory for data and histogram
    int *data = (int *)malloc(sizeof(int) * SIZE);
    int *histogram = (int *)calloc(RANGE, sizeof(int));  // initializes to zero
    if (data == NULL || histogram == NULL) {
        printf("Memory allocation failed\n");
        return 1; // exit if malloc or calloc fails
    }

    // Seed random number generator for different results each run
    srand(time(NULL));

    // Generate random data
    printf("Generated Data:\n");
    for (int i = 0; i < SIZE; i++) {
        data[i] = rand() % RANGE;
        printf("%d ", data[i]);
    }
    printf("\n");

    // Measure time
    double t0 = get_clock();
    // Compute histogram
    computeHistogram(data, histogram, SIZE, RANGE);
    double t1 = get_clock();

    // Print the histogram
    printf("Histogram:\n");
    for (int i = 0; i < RANGE; i++) {
        printf("%d: %d\n", i, histogram[i]);
    }

    // Print time taken
    printf("Time: %f ns\n", 1000000000.0 * (t1 - t0));

    // Free allocated memory
    free(data);
    free(histogram);

    return 0;
}
