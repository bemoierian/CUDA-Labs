// Bemoi_Erian_Ayad_Sec1_BN17
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[])
{
    // If the input is not valid, set the default input to the test case given in the requirement
    if (argc < 3 || argc < 3 + atoi(argv[1]) * atoi(argv[2]))
    {
        char *new_argv[] = {"program_name", "3", "3", "10", "20", "30", "5", "10", "20", "2", "4", "6", NULL};
        argv = new_argv;
    }

    // List dimensions
    int nrows = atoi(argv[1]);
    int ncols = atoi(argv[2]);
    // Creating a 2D list
    int **list_2D;
    list_2D = (int **)malloc(nrows * sizeof(int *));
    for (int i = 0; i < nrows; i++)
    {
        list_2D[i] = (int *)malloc(ncols * sizeof(int));
    }
    // Filling the list
    int inputI = 3;
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            list_2D[i][j] = atoi(argv[inputI++]);
        }
    }
    // calculating the sum
    int sum = 0;
    // Looping through the columns
    for (int j = 0; j < ncols; j++)
    {
        // Temp var to store the concatenated number of each column
        // Initialize it with the first element of the column
        int tempSum = list_2D[0][j];
        // Looping through the rows
        for (int i = 1; i < nrows; i++)
        {
            // Convert the number to string to get the number of digits
            // Buffer for 11 digit number
            char str[12];
            sprintf(str, "%d", list_2D[i][j]);
            int length = strlen(str);
            tempSum = tempSum * pow(10, length) + list_2D[i][j];
        }
        // Add the concatenated number of each column to the sum
        sum += tempSum;
    }
    printf("%d", sum);
    // Deallocating memory
    for (int i = 0; i < nrows; i++)
    {
        free(list_2D[i]);
    }
    free(list_2D);

    return 0;
}
