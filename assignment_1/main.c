#include <stdio.h>
#include "average.h"


int main() {
    double arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    double result = average(5, arr);

    printf("I added 5.0 to the array and updated the average. The average of 1, 2, 3, 4 and 5 is: %.4f\n", result);
    return 0;    
}

