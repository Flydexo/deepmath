#include <stdio.h>
#include <stdlib.h>
#include "read-image.h"

// Read MNIST images
void load_mnist_images(const char *filename, unsigned char **data, int num_images_to_read)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        printf("Failed to open file: %s\n", filename);
        return;
    }

    // Read MNIST header
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // Read magic number
    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = __builtin_bswap32(magic_number);

    // Read number of images
    fread(&number_of_images, sizeof(int), 1, fp);
    number_of_images = __builtin_bswap32(number_of_images);

    // Read number of rows
    fread(&n_rows, sizeof(int), 1, fp);
    n_rows = __builtin_bswap32(n_rows);

    // Read number of columns
    fread(&n_cols, sizeof(int), 1, fp);
    n_cols = __builtin_bswap32(n_cols);

    // Make sure we don't try to read more images than available
    if (num_images_to_read > number_of_images)
    {
        num_images_to_read = number_of_images;
        printf("Warning: Reduced number of images to read to %d\n", num_images_to_read);
    }

    // Allocate memory for multiple images
    int image_size = n_rows * n_cols;
    *data = malloc(image_size * num_images_to_read);
    if (!*data)
    {
        printf("Failed to allocate memory\n");
        fclose(fp);
        return;
    }

    // Read multiple images
    fread(*data, sizeof(unsigned char), image_size * num_images_to_read, fp);

    fclose(fp);
}

int main_image()
{
    unsigned char *data;
    int num_images_to_read = 10; // Change this to read more or fewer images
    load_mnist_images("train-images.idx3-ubyte", &data, num_images_to_read);

    // Print each image
    for (int img = 0; img < num_images_to_read; img++)
    {
        printf("\nImage %d:\n", img + 1);
        for (int i = 0; i < 784; ++i)
        {
            if (i % 28 == 0)
                printf("\n");
            // Use ASCII characters to represent different intensities
            char c = ' ';
            int pixel = data[img * 784 + i];
            if (pixel > 200)
                c = '#';
            else if (pixel > 150)
                c = '+';
            else if (pixel > 100)
                c = '.';
            printf("%c", c);
        }
        printf("\n");
    }

    free(data);
    return 0;
}