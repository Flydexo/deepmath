#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include "../micrograd/micrograd.h"
#include "read-image.h"
#include "read-label.h"
#define WEIGHTS_LEN 8 * 785 + 8 * 9 + 10 * 9

typedef struct
{
    Value **weights;
    Value *(*activation)(int, Value **, int);
    int len;
} Neuron;

typedef struct
{
    int len;
    Value **coeffs;
} Vect;

typedef struct
{
    int len;
    Neuron *neurons;
} Layer;

typedef struct
{
    int len;
    Layer *layers;
} Model;

Vect predict(Model *model, Vect *input)
{
    Vect data = *input;
    for (int i = 0; i < model->len; i++)
    {
        Layer layer = model->layers[i];
        Vect next_data = {layer.len, malloc(sizeof(float) * layer.len)};
        for (int j = 0; j < layer.len; j++)
        {
            Neuron neuron = layer.neurons[j];
            Value *x = v_init(0.0);
            for (int k = 0; k < data.len; k++)
            {
                x = v_add(x, v_mult(neuron.weights[k], data.coeffs[k]));
            }
            x = v_add(x, neuron.weights[data.len]); // add bias
            next_data.coeffs[j] = x;
        }
        for (int j = 0; j < layer.len; j++)
        {
            data.coeffs[j] = layer.neurons[0].activation(j, next_data.coeffs, layer.len);
        }
        data.len = layer.len;
    }
    return data;
}

Neuron make_neuron(int len, Value *(*activation)(int, Value **, int))
{
    Value **weights = malloc(sizeof(Value *) * len);
    float scale = sqrt(2.0 / len); // He initialization
    for (int i = 0; i < len; i++)
    {
        weights[i] = v_init(((float)rand() / RAND_MAX * 2 - 1) * scale);
    }
    Neuron neuron = {weights, activation, len};
    return neuron;
}

Layer make_layer(int p, Value *(*activation)(int, Value **, int), int prev_p)
{
    Layer layer = {p, malloc(sizeof(Neuron) * p)};
    for (int i = 0; i < p; i++)
    {
        layer.neurons[i] = make_neuron(prev_p + 1, activation);
    }
    return layer;
}

Model make_mnsit_model()
{
    Model model = {3, malloc(sizeof(Layer) * 3)};
    model.layers[0] = make_layer(8, v_sigmoid, 784);
    model.layers[1] = make_layer(8, v_sigmoid, 8);
    model.layers[2] = make_layer(10, v_softmax, 8);
    return model;
}

Value *loss(Model *model, Vect *input, Vect *expected_output)
{
    Vect output = predict(model, input);
    Value *neg = v_init(-1);
    Value *l = v_init(0);
    for (int i = 0; i < expected_output->len; i++)
    {
        v_add(l, v_mult(v_mult(expected_output->coeffs[i], v_log(output.coeffs[i])), neg));
    }
    return l;
}

float acc(Model *model)
{
    unsigned char *data;
    unsigned char *labels;
    load_mnist_images("train-images.data", &data, 10000);
    load_mnist_labels("train-labels.data", &labels, 10000);
    int correct = 0;
    printf("Random weight from layer 0: %f\n", model->layers[0].neurons[5].weights[242]->data);
    for (int idx = 0; idx < 10000; idx++)
    {
        // Create input vector from image data
        // Print a random weight from layer 0
        Vect input = {784, malloc(sizeof(float) * 784)};
        for (int j = 0; j < 784; j++)
        {
            input.coeffs[j] = v_init((float)data[idx * 784 + j]);
        }

        // Get prediction
        Vect output = predict(model, &input);

        // Find max probability index
        int pred = 0;
        float max_prob = output.coeffs[0]->data;
        for (int j = 1; j < output.len; j++)
        {
            if (output.coeffs[j]->data > max_prob)
            {
                max_prob = output.coeffs[j]->data;
                pred = j;
            }
        }

        // Compare with true label
        if (pred == labels[idx])
        {
            correct++;
        }

        free(input.coeffs);

        for (int j = 0; j < output.len; j++)
        {
            free(output.coeffs[j]);
        }
    }

    return (float)correct /
           100.0;
}

Vect categorize(char output)
{
    Vect a = {10, malloc(sizeof(Value *) * 10)};
    for (int i = 0; i < 10; i++)
    {
        if (i == output)
        {
            a.coeffs[i] = v_init(1);
        }
        else
        {
            a.coeffs[i] = v_init(0);
        }
    }
    return a;
}

void update_weights(Model *model, Value *delta)
{
    int nb_weights = 0;
    for (int i = 0; i < model->len; i++)
    {
        if (i > 0)
        {
            nb_weights = model->layers[i - 1].len + 1;
        }
        for (int j = 0; j < model->layers[i].len; j++)
        {
            for (int l = 0; l < nb_weights; l++)
            {
                model->layers[i].neurons[j].weights[l] = v_add(model->layers[i].neurons[j].weights[l], v_mult(v_init(-1), v_mult(delta, v_init(model->layers[i].neurons[j].weights[l]->grad))));
            }
        }
    }
}

void retro(Model *model, int nmax, Value *delta)
{
    unsigned char *data;
    unsigned char *labels;
    load_mnist_images("train-images.data", &data, 60000);
    load_mnist_labels("train-labels.data", &labels, 60000);

    // Start with a smaller number of samples for testing
    int test_samples = 1000;
    printf("Initial accuracy on %d samples:\n", test_samples);
    float initial_acc = acc(model);

    for (int i = 0; i < nmax; i++)
    {
        Vect input = {784, malloc(sizeof(float) * 784)};
        for (int j = 0; j < 784; j++)
        {
            input.coeffs[j] = v_init((double)data[i * 784 + j] / 255.0);
        }

        if (i % 100 == 0)
        {
            Vect output = predict(model, &input);
            printf("\nSample %d (label: %d) - Predictions: ", i, labels[i]);
            for (int j = 0; j < output.len; j++)
            {
                printf("%.4f ", output.coeffs[j]->data);
            }
            printf("\n");

            float current_acc = acc(model);
            printf("Accuracy: %.2f%% (change: %.2f%%)\n",
                   current_acc, current_acc - initial_acc);
            for (int j = 0; j < output.len; j++)
            {
                free(output.coeffs[j]);
            }
        }
        Vect output = categorize(labels[i]);
        v_backward(loss(model, &input, &output)); // compute the gradiants for each weight
        update_weights(model, delta);
        free(input.coeffs);
    }
}

int main()
{
    srand(time(NULL));
    Model model = make_mnsit_model();
    printf("\n\n");
    printf(".___________..______          ___       __  .__   __.  __  .__   __.   _______ \n");
    printf("|           ||   _  \\        /   \\     |  | |  \\ |  | |  | |  \\ |  |  /  _____|\n");
    printf("`---|  |----`|  |_)  |      /  ^  \\    |  | |   \\|  | |  | |   \\|  | |  |  __  \n");
    printf("    |  |     |      /      /  /_\\  \\   |  | |  . `  | |  | |  . `  | |  | |_ | \n");
    printf("    |  |     |  |\\  \\----./  _____  \\  |  | |  |\\   | |  | |  |\\   | |  |__| | \n");
    printf("    |__|     | _| `._____/__/     \\__\\ |__| |__| \\__| |__| |__| \\__|  \\______| \n");
    printf("                                                                               \n");
    retro(&model, 10000, v_init(0.01)); // Reduced number of iterations and learning rate
    return 0;
}