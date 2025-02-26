#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include "read-image.h"
#include "read-label.h"
#define WEIGHTS_LEN 8 * 785 + 8 * 9 + 10 * 9
#include "stdbool.h"

float sum(float acc, float *values, int length)
{
    if (length == 0)
        return acc;
    return sum(acc + values[0], &values[1], length - 1);
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
float sigmoid(int i, float *x, int _)
{
    return 1 / (1 + exp(-x[i]));
}

float d_sigmoid(int i, float *x, int length)
{
    return sigmoid(i, x, length) * (1.0 - sigmoid(i, x, length));
}

float *apply_f(float *x, int length, float (*f)(float x))
{
    float *images = malloc(sizeof(float) * length);
    for (int i = 0; i < length; i++)
    {
        images[i] = f(x[i]);
    }
    return images;
}

float softmax(int i, float *x, int length)
{
    // Find maximum value for numerical stability
    float max_val = x[0];
    for (int j = 1; j < length; j++)
    {
        if (x[j] > max_val)
            max_val = x[j];
    }

    // Compute exp(x - max) for numerical stability
    float *y = malloc(sizeof(float) * length);
    float sum = 0.0;
    for (int j = 0; j < length; j++)
    {
        y[j] = exp(x[j] - max_val);
        sum += y[j];
    }

    // Compute final result
    float result = y[i] / sum;

    free(y);
    return result;
}

float kronecker(int i, int j)
{
    if (i == j)
        return 1.0;
    else
        return 0.0;
}

float d_softmax(int i, float *x, int length, int di)
{
    return softmax(i, x, length) * (kronecker(i, di) - softmax(di, x, length));
}

float cross_entropy(float x)
{
    return -log(x);
}

typedef struct
{
    bool initialized;
    float value;
} d;

typedef struct
{
    float value;
    d grad;
} Weight;

typedef struct
{
    Weight *weights;
    float (*activation)(int, float *, int);
} Neuron;

typedef struct
{
    int len;
    float *coeffs;
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
            float x = 0.0;
            for (int k = 0; k < data.len; k++)
            {
                x += neuron.weights[k].value * data.coeffs[k];
            }
            x += neuron.weights[data.len].value; // add bias
            next_data.coeffs[j] = x;
        }
        // activation (needed after loop for softmax)
        for (int j = 0; j < layer.len; j++)
        {
            data.coeffs[j] = layer.neurons[0].activation(j, next_data.coeffs, layer.len);
        }
        data.len = layer.len;
        // printf("Layer length: %d\n", layer.len);
    }
    return data;
}

Neuron make_neuron(int len, float (*activation)(int, float *, int))
{
    Weight *weights = malloc(sizeof(Weight) * len);
    float scale = sqrt(2.0 / len); // He initialization
    for (int i = 0; i < len; i++)
    {
        weights[i].value = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    }
    Neuron neuron = {weights, activation};
    return neuron;
}

Layer make_layer(int p, float (*activation)(int, float *, int), int prev_p)
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
    model.layers[0] = make_layer(8, sigmoid, 784);
    model.layers[1] = make_layer(8, sigmoid, 8);
    model.layers[2] = make_layer(10, softmax, 8);
    return model;
}

float derivate(Model *model, Vect *input, int layer_i, int neuron_i, int weight_i)
{
    if (model->layers[layer_i].neurons[neuron_i].weights[weight_i].grad.initialized)
    {
        return model->layers[layer_i].neurons[neuron_i].weights[weight_i].grad.value;
    }
    Layer layer = model->layers[layer_i];
    Neuron neuron = layer.neurons[neuron_i];
    float weight = neuron.weights[weight_i].value;
    if (layer_i == model->len - 1)
    {
        // For the last layer (softmax)
        Vect output = predict(model, input);

        // Get the input to this layer
        Model prev_model = {layer_i, model->layers};
        Vect prev_output = predict(&prev_model, input);

        // If this is the bias weight
        if (weight_i == prev_output.len)
        {
            return d_softmax(neuron_i, output.coeffs, layer.len, neuron_i);
        }

        // For regular weights
        return d_softmax(neuron_i, output.coeffs, layer.len, neuron_i) * prev_output.coeffs[weight_i];
    }
    else
    {
        // For hidden layers (sigmoid)
        Model prev_model = {layer_i, model->layers};
        float f_star = predict(&prev_model, input).coeffs[weight_i];
        float axf_star = weight * f_star;
        float acc = 0.0;
        Layer next_layer = model->layers[layer_i + 1];
        for (int i = 0; i < next_layer.len; i++)
        {
            float w = next_layer.neurons[i].weights[neuron_i].value;
            acc += f_star * d_sigmoid(0, &axf_star, 1) / sigmoid(0, &axf_star, 1) * w * derivate(model, input, layer_i + 1, i, neuron_i);
        }
        model->layers[layer_i].neurons[neuron_i].weights[weight_i].grad.value = acc;
        model->layers[layer_i].neurons[neuron_i].weights[weight_i].grad.initialized = true;
        return acc;
    }
}

void reset_memory(Model *model)
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
                model->layers[i].neurons[j].weights[l].grad.initialized = false;
            }
        }
    }
}

Vect grad(Model *model, Vect *input, int expected, float delta)
{
    Vect output = predict(model, input);
    Vect total_grad = {WEIGHTS_LEN, malloc(sizeof(float) * WEIGHTS_LEN)};
    int k = 0;
    int nb_weights = 785;

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
                float grad_val = derivate(model, input, i, j, l);
                total_grad.coeffs[k] = delta * grad_val;
                k++;
            }
        }
    }
    reset_memory(model);
    return total_grad;
}

float f_minus(float a, float b)
{
    return a - b;
}

void self_update_weights(Model *model, float (*f)(float, float), Vect b)
{
    int nb_weights = 0;
    int k = 0;
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
                if (!isnan(b.coeffs[k]))
                {
                    model->layers[i].neurons[j].weights[l].value = f(model->layers[i].neurons[j].weights[l].value, b.coeffs[k]);
                }
                k++;
            }
        }
    }
}

float acc(Model *model)
{
    unsigned char *data;
    unsigned char *labels;
    load_mnist_images("train-images", &data, 10000);
    load_mnist_labels("train-labels", &labels, 10000);
    int correct = 0;
    printf("Random weight from layer 0: %f\n", model->layers[0].neurons[5].weights[242].value);
    for (int idx = 0; idx < 10000; idx++)
    {
        // Create input vector from image data
        // Print a random weight from layer 0
        Vect input = {784, malloc(sizeof(float) * 784)};
        for (int j = 0; j < 784; j++)
        {
            input.coeffs[j] = (float)data[idx * 784 + j];
        }

        // Get prediction
        Vect output = predict(model, &input);

        // Find max probability index
        int pred = 0;
        float max_prob = output.coeffs[0];
        for (int j = 1; j < output.len; j++)
        {
            if (output.coeffs[j] > max_prob)
            {
                max_prob = output.coeffs[j];
                pred = j;
            }
        }

        // Compare with true label
        if (pred == labels[idx])
        {
            correct++;
        }

        free(input.coeffs);
    }

    return (float)correct / 100.0;
}

void retro(Model *model, int nmax, float delta)
{
    unsigned char *data;
    unsigned char *labels;
    load_mnist_images("train-images", &data, 60000);
    load_mnist_labels("train-labels", &labels, 60000);

    // Start with a smaller number of samples for testing
    int test_samples = 1000;
    printf("Initial accuracy on %d samples:\n", test_samples);
    float initial_acc = acc(model);

    for (int i = 0; i < nmax; i++)
    {
        Vect input = {784, malloc(sizeof(float) * 784)};
        for (int j = 0; j < 784; j++)
        {
            input.coeffs[j] = (float)data[i * 784 + j] / 255.0;
        }

        if (i % 100 == 0)
        {
            Vect output = predict(model, &input);
            printf("\nSample %d (label: %d) - Predictions: ", i, labels[i]);
            for (int j = 0; j < output.len; j++)
            {
                printf("%.4f ", output.coeffs[j]);
            }
            printf("\n");

            float current_acc = acc(model);
            printf("Accuracy: %.2f%% (change: %.2f%%)\n",
                   current_acc, current_acc - initial_acc);
        }

        Vect gradv = grad(model, &input, (int)labels[i], delta);
        self_update_weights(model, f_minus, gradv);
        free(input.coeffs);
    }
}

int main()
{
    srand(time(NULL));
    Model model = make_mnsit_model();
    Vect input = {784, malloc(sizeof(float) * 784)};
    printf("Input: ");
    for (int i = 0; i < 784; i++)
    {
        input.coeffs[i] = (float)rand() / RAND_MAX * 255;
        printf("%f, ", input.coeffs[i]);
    }
    printf("\n\n\n=================================\n\n\n\n");
    Vect output = predict(&model, &input);
    printf("Output: ");
    for (int i = 0; i < output.len; i++)
    {
        printf("%f, ", output.coeffs[i]);
    }
    printf("\n\n");
    printf(".___________..______          ___       __  .__   __.  __  .__   __.   _______ \n");
    printf("|           ||   _  \\        /   \\     |  | |  \\ |  | |  | |  \\ |  |  /  _____|\n");
    printf("`---|  |----`|  |_)  |      /  ^  \\    |  | |   \\|  | |  | |   \\|  | |  |  __  \n");
    printf("    |  |     |      /      /  /_\\  \\   |  | |  . `  | |  | |  . `  | |  | |_ | \n");
    printf("    |  |     |  |\\  \\----./  _____  \\  |  | |  |\\   | |  | |  |\\   | |  |__| | \n");
    printf("    |__|     | _| `._____/__/     \\__\\ |__| |__| \\__| |__| |__| \\__|  \\______| \n");
    printf("                                                                               \n");
    retro(&model, 10000, 0.001); // Reduced number of iterations and learning rate
    return 0;
}