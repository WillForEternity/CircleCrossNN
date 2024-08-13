// See `CircleCrossnnLRdecay.c` for a fully commented version that doesn't have a command line interface

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_LAYERS 10
#define MAX_NEURONS_PER_LAYER 100
#define INPUT_SIZE 784

// Tunable parameters (now global variables)
int NUM_TRAINING_EXAMPLES = 1000;
int EPOCHS = 50;
double INITIAL_LEARNING_RATE = 0.9;
double DECAY_RATE = 0.15;
double MIN_LEARNING_RATE = 0.08;

typedef struct {
    int num_layers;
    int layer_sizes[MAX_LAYERS];
    double **weights;
    double **biases;
    double **activations;
    double **z_values;
} NeuralNetwork;

// Function declarations
double adjust_learning_rate(double current_rate, int epoch, double error);
void print_image(double* image);
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes);
void free_network(NeuralNetwork* nn);
void forward_propagation(NeuralNetwork* nn, double* input);
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate);
void generate_image(double* image, int* label);
void* safe_malloc(size_t size);
double sigmoid(double x);
double sigmoid_derivative(double x);
void draw_line(double* image, double m, double b, int thickness);
void tune_parameters();
void run_neural_network();

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

void* safe_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

double adjust_learning_rate(double current_rate, int epoch, double error) {
    double new_rate = INITIAL_LEARNING_RATE / (1 + DECAY_RATE * epoch);
    if (new_rate < MIN_LEARNING_RATE) {
        new_rate = MIN_LEARNING_RATE;
    }
    return new_rate;
}

void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes) {
    if (num_layers < 2 || num_layers > MAX_LAYERS) {
        fprintf(stderr, "Invalid number of layers\n");
        exit(EXIT_FAILURE);
    }

    nn->num_layers = num_layers;
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    nn->weights = safe_malloc((num_layers - 1) * sizeof(double*));
    nn->biases = safe_malloc((num_layers - 1) * sizeof(double*));
    nn->activations = safe_malloc(num_layers * sizeof(double*));
    nn->z_values = safe_malloc(num_layers * sizeof(double*));

    for (int i = 0; i < num_layers - 1; i++) {
        nn->weights[i] = safe_malloc(layer_sizes[i] * layer_sizes[i+1] * sizeof(double));
        nn->biases[i] = safe_malloc(layer_sizes[i+1] * sizeof(double));
        
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i+1]; j++) {
            nn->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        
        for (int j = 0; j < layer_sizes[i+1]; j++) {
            nn->biases[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    for (int i = 0; i < num_layers; i++) {
        nn->activations[i] = safe_malloc(layer_sizes[i] * sizeof(double));
        nn->z_values[i] = safe_malloc(layer_sizes[i] * sizeof(double));
    }
}

void free_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        free(nn->weights[i]);
        free(nn->biases[i]);
    }

    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        free(nn->z_values[i]);
    }

    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);
}

void forward_propagation(NeuralNetwork* nn, double* input) {
    memcpy(nn->activations[0], input, nn->layer_sizes[0] * sizeof(double));

    for (int i = 1; i < nn->num_layers; i++) {
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            double sum = nn->biases[i-1][j];
            
            for (int k = 0; k < nn->layer_sizes[i-1]; k++) {
                sum += nn->activations[i-1][k] * nn->weights[i-1][k * nn->layer_sizes[i] + j];
            }
            
            nn->z_values[i][j] = sum;
            nn->activations[i][j] = sigmoid(sum);
        }
    }
}

void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate) {
    int output_layer = nn->num_layers - 1;
    
    double* delta = safe_malloc(nn->layer_sizes[output_layer] * sizeof(double));
    
    for (int i = 0; i < nn->layer_sizes[output_layer]; i++) {
        double error = nn->activations[output_layer][i] - target[i];
        delta[i] = error * sigmoid_derivative(nn->activations[output_layer][i]);
    }

    for (int l = output_layer - 1; l >= 0; l--) {
        double* prev_delta = safe_malloc(nn->layer_sizes[l] * sizeof(double));
        
        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            double error = 0.0;
            for (int j = 0; j < nn->layer_sizes[l+1]; j++) {
                error += delta[j] * nn->weights[l][i * nn->layer_sizes[l+1] + j];
            }
            prev_delta[i] = error * sigmoid_derivative(nn->activations[l][i]);
        }

        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            for (int j = 0; j < nn->layer_sizes[l+1]; j++) {
                nn->weights[l][i * nn->layer_sizes[l+1] + j] -= learning_rate * nn->activations[l][i] * delta[j];
            }
        }
        
        for (int i = 0; i < nn->layer_sizes[l+1]; i++) {
            nn->biases[l][i] -= learning_rate * delta[i];
        }

        free(delta);
        delta = prev_delta;
    }

    free(delta);
}

void draw_line(double* image, double m, double b, int thickness) {
    for (int x = 0; x < 28; x++) {
        int y = round(m * x + b);
        
        for (int dx = -thickness/2; dx <= thickness/2; dx++) {
            for (int dy = -thickness/2; dy <= thickness/2; dy++) {
                int px = x + dx;
                int py = y + dy;
                
                if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                    image[py * 28 + px] = 1.0;
                }
            }
        }
    }
}

void generate_image(double* image, int* label) {
    memset(image, 0, INPUT_SIZE * sizeof(double));

    int thickness = 1 + rand() % 3;
    
    *label = rand() % 2;
    
    if (*label == 0) {  // Generate a cross
        int ix = 4 + rand() % 20;
        int iy = 4 + rand() % 20;
        
        double m1 = (double)(rand() % 200 - 100) / 100;
        double m2;
        
        do {
            m2 = (double)(rand() % 200 - 100) / 100;
        } while (fabs(m2 - m1) < 0.1);

        double b1 = iy - m1 * ix;
        double b2 = iy - m2 * ix;

        draw_line(image, m1, b1, thickness);
        draw_line(image, m2, b2, thickness);
    } else {  // Generate a circle
        int radius = 5 + rand() % 8;
        
        int center_x = rand() % 28;
        int center_y = rand() % 28;
        
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                double distance = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
                
                if (fabs(distance - radius) < thickness / 2.0) {
                    image[y * 28 + x] = 1.0;
                }
            }
        }
    }
}

void print_image(double* image) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%c", image[i * 28 + j] > 0.5 ? '#' : '.');
        }
        printf("\n");
    }
}

void tune_parameters() {
    printf("Current parameters:\n");
    printf("1. Number of training examples: %d\n", NUM_TRAINING_EXAMPLES);
    printf("2. Number of epochs: %d\n", EPOCHS);
    printf("3. Initial learning rate: %.2f\n", INITIAL_LEARNING_RATE);
    printf("4. Decay rate: %.2f\n", DECAY_RATE);
    printf("5. Minimum learning rate: %.2f\n", MIN_LEARNING_RATE);

    printf("\nDo you want to change any parameters? (y/n): ");
    char choice;
    scanf(" %c", &choice);

    if (choice == 'y' || choice == 'Y') {
        printf("\nEnter new values for the parameters (or press enter to keep current value):\n");
        
        char input[20];
        int temp_int;
        double temp_double;

        printf("\n");
        printf("    Number of training examples: ");
        fgets(input, sizeof(input), stdin);  // Clear input buffer
        fgets(input, sizeof(input), stdin);
        if (sscanf(input, "%d", &temp_int) == 1) NUM_TRAINING_EXAMPLES = temp_int;

        printf("    Number of epochs: ");
        fgets(input, sizeof(input), stdin);
        if (sscanf(input, "%d", &temp_int) == 1) EPOCHS = temp_int;

        printf("    Initial learning rate: ");
        fgets(input, sizeof(input), stdin);
        if (sscanf(input, "%lf", &temp_double) == 1) INITIAL_LEARNING_RATE = temp_double;

        printf("    Decay rate: ");
        fgets(input, sizeof(input), stdin);
        if (sscanf(input, "%lf", &temp_double) == 1) DECAY_RATE = temp_double;

        printf("    Minimum learning rate: ");
        fgets(input, sizeof(input), stdin);
        if (sscanf(input, "%lf", &temp_double) == 1) MIN_LEARNING_RATE = temp_double;

        printf("\nUpdated parameters:\n");
        printf("1. Number of training examples: %d\n", NUM_TRAINING_EXAMPLES);
        printf("2. Number of epochs: %d\n", EPOCHS);
        printf("3. Initial learning rate: %.2f\n", INITIAL_LEARNING_RATE);
        printf("4. Decay rate: %.2f\n", DECAY_RATE);
        printf("5. Minimum learning rate: %.2f\n", MIN_LEARNING_RATE);
    }
}

void run_neural_network() {
    srand(time(NULL));

    int num_layers = 5;
    int layer_sizes[] = {INPUT_SIZE, 64, 20, 20, 2};

    NeuralNetwork nn;
    init_network(&nn, num_layers, layer_sizes);

    double learning_rate = INITIAL_LEARNING_RATE;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;
        double total_error = 0.0;

        for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
            double input[INPUT_SIZE];
            int label;
            generate_image(input, &label);

            forward_propagation(&nn, input);

            double target[2] = {0, 0};
            target[label] = 1;

            double example_error = 0.0;
            for (int j = 0; j < 2; j++) {
                double diff = nn.activations[num_layers-1][j] - target[j];
                example_error += diff * diff;
            }
            total_error += example_error;

            backward_propagation(&nn, target, learning_rate);

            int predicted = nn.activations[num_layers-1][0] > nn.activations[num_layers-1][1] ? 0 : 1;
            if (predicted == label) correct++;
        }

        double avg_error = total_error / NUM_TRAINING_EXAMPLES;

        learning_rate = adjust_learning_rate(learning_rate, epoch, avg_error);

        printf("Epoch %d, Accuracy: %.2f%%, Avg Error: %.4f, Learning Rate: %.4f\n", 
               epoch, (float)correct / NUM_TRAINING_EXAMPLES * 100, avg_error, learning_rate);
    }

    printf("\n\nTesting the neural network:\n\n");
    int correct = 0;
    for (int i = 0; i < 30; i++) {
        double input[INPUT_SIZE];
        int label;
        generate_image(input, &label);

        forward_propagation(&nn, input);

        int predicted = nn.activations[num_layers-1][0] > nn.activations[num_layers-1][1] ? 0 : 1;

        if (predicted == label) correct++;

        printf("Actual: %s, Predicted: %s\n", 
               label == 0 ? "Cross" : "Circle", 
               predicted == 0 ? "Cross" : "Circle");
        print_image(input);
        printf("\n");
    }

    printf("Test Accuracy: %.2f%%\n", (float)correct / 30 * 100);

    free_network(&nn);
}

int main() {
    printf("\nWelcome to TuneNN, the neural network that you can tune from the command line interface!\n\n");

    tune_parameters();

    printf("\nRunning the neural network with the current parameters...\n\n");
    run_neural_network();

    return 0;
}
