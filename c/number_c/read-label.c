#include <stdio.h>
#include <stdlib.h>
#include "read-label.h"

// Read MNIST labels
void load_mnist_labels(const char *filename, unsigned char **labels, int num_labels_to_read)
{
    // Initialize labels pointer to NULL
    *labels = NULL;

    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    // Read MNIST header
    int magic_number = 0;
    int number_of_labels = 0;

    // Read magic number
    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = __builtin_bswap32(magic_number);

    // Read number of labels
    fread(&number_of_labels, sizeof(int), 1, fp);
    number_of_labels = __builtin_bswap32(number_of_labels);

    // Make sure we don't try to read more labels than available
    if (num_labels_to_read > number_of_labels)
    {
        num_labels_to_read = number_of_labels;
        printf("Warning: Reduced number of labels to read to %d\n", num_labels_to_read);
    }

    // Allocate memory for labels
    *labels = malloc(num_labels_to_read);
    if (!*labels)
    {
        printf("Failed to allocate memory\n");
        fclose(fp);
        return;
    }

    // Read labels
    fread(*labels, sizeof(unsigned char), num_labels_to_read, fp);

    fclose(fp);
}

int main_label()
{
    unsigned char *labels = NULL; // Initialize to NULL
    int num_labels_to_read = 100;
    load_mnist_labels("train-labels.idx1-ubyte", &labels, num_labels_to_read);

    // Only process and free if labels were successfully loaded
    if (labels)
    {
        for (int i = 0; i < num_labels_to_read; i++)
        {
            printf("Label %d: %d\n", i + 1, labels[i]);
        }
        free(labels);
    }

    return 0;
}
