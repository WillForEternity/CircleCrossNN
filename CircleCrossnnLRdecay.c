// Header includes
#include <stdio.h>   // Standard input/output operations
#include <stdlib.h>  // General utilities: memory allocation, random numbers, etc.
#include <math.h>    // Mathematical functions like exp() for sigmoid
#include <time.h>    // Time-related functions, used for seeding random number generator
#include <string.h>  // String manipulation functions like memcpy()

// Constants for network architecture and training
#define MAX_LAYERS 10                // Maximum number of layers in the neural network
#define MAX_NEURONS_PER_LAYER 100    // Maximum neurons in any single layer
#define INPUT_SIZE 784               // Size of input layer (28x28 pixels)

#define NUM_TRAINING_EXAMPLES 1000   // Number of training examples per epoch
#define EPOCHS 50                    // Number of training epochs

// Learning rate parameters
#define INITIAL_LEARNING_RATE 0.9    // Starting learning rate
#define DECAY_RATE 0.15               // Rate at which learning rate decays
#define MIN_LEARNING_RATE 0.08       // Minimum learning rate

// Neural Network structure definition
typedef struct {
    int num_layers;                          // Total number of layers in the network
    int layer_sizes[MAX_LAYERS];             // Array to store the size of each layer
    double **weights;                        // 2D array for weights between layers
    double **biases;                         // 2D array for biases of each neuron
    double **activations;                    // 2D array for neuron activations
    double **z_values;                       // 2D array for weighted sums before activation
} NeuralNetwork;

// function declarations
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


// sigmoid function: 
// as x approaches positive infinity, sigmoid(x) approaches 1
// as x approaches negative infinity,sigmoid(x) approraches 0
// sigmoid(0) = 1/(1+1) = 1/2
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// full differentiation shown on Github
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Safe memory allocation function
void* safe_malloc(size_t size) {
    // Allocate memory of the specified size
    void* ptr = malloc(size);

    // Check if memory allocation was successful
    if (ptr == NULL) {
        // If allocation failed, print an error message and exit the program
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Return the pointer to the allocated memory
    return ptr;
}

// Adjust the learning rate based on the current epoch and error
double adjust_learning_rate(double current_rate, int epoch, double error) {
    // Calculate the new learning rate using the decay formula
    double new_rate = INITIAL_LEARNING_RATE / (1 + DECAY_RATE * epoch);

    // Ensure the new learning rate does not fall below the minimum learning rate
    if (new_rate < MIN_LEARNING_RATE) {
        new_rate = MIN_LEARNING_RATE;
    }

    // Return the adjusted learning rate
    return new_rate;
}

// Function to initialize the neural network
// Parameters:
//   nn: pointer to the NeuralNetwork structure to be initialized
//   num_layers: total number of layers in the network
//   layer_sizes: array containing the number of neurons in each layer
void init_network(NeuralNetwork* nn, int num_layers, int* layer_sizes) {
    // Check if the number of layers is valid (at least 2 layers and not exceeding MAX_LAYERS)
    if (num_layers < 2 || num_layers > MAX_LAYERS) {
        // If invalid, print error message to stderr and exit the program
        fprintf(stderr, "Invalid number of layers\n");
        exit(EXIT_FAILURE);
    }

    // Set the number of layers in the network structure
    nn->num_layers = num_layers;
    // Copy the layer sizes array into the network structure
    // This uses memcpy to efficiently copy 'num_layers' integers
    memcpy(nn->layer_sizes, layer_sizes, num_layers * sizeof(int));

    // Allocate memory for pointers to weights between layers
    // There are (num_layers - 1) sets of weights, as weights connect adjacent layers
    nn->weights = safe_malloc((num_layers - 1) * sizeof(double*));
    // Allocate memory for pointers to biases for each layer (except input layer)
    nn->biases = safe_malloc((num_layers - 1) * sizeof(double*));
    // Allocate memory for pointers to activation values for each layer
    nn->activations = safe_malloc(num_layers * sizeof(double*));
    // Allocate memory for pointers to z-values (weighted sums before activation) for each layer
    nn->z_values = safe_malloc(num_layers * sizeof(double*));

    // Loop through each layer (except the last one) to initialize weights and biases
    for (int i = 0; i < num_layers - 1; i++) {
        // Allocate memory for weights between current layer i and next layer i+1
        // The number of weights is the product of neurons in current and next layer
        nn->weights[i] = safe_malloc(layer_sizes[i] * layer_sizes[i+1] * sizeof(double));
        // Allocate memory for biases of the next layer
        nn->biases[i] = safe_malloc(layer_sizes[i+1] * sizeof(double));
        
        // Initialize each weight with a random value between -1 and 1
        for (int j = 0; j < layer_sizes[i] * layer_sizes[i+1]; j++) {
            // rand() / RAND_MAX gives a value between 0 and 1
            // Multiply by 2 and subtract 1 to get a value between -1 and 1
            nn->weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        
        // Initialize each bias of the next layer with a random value between -1 and 1
        for (int j = 0; j < layer_sizes[i+1]; j++) {
            nn->biases[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    // Allocate memory for activation values and z-values for each layer
    for (int i = 0; i < num_layers; i++) {
        // Allocate memory for activation values of current layer
        nn->activations[i] = safe_malloc(layer_sizes[i] * sizeof(double));
        // Allocate memory for z-values of current layer
        nn->z_values[i] = safe_malloc(layer_sizes[i] * sizeof(double));
    }
}

// Function to free all dynamically allocated memory for the neural network
// Parameter:
//   nn: pointer to the NeuralNetwork structure to be freed
void free_network(NeuralNetwork* nn) {
    // Loop through each layer except the last one
    // This is because weights and biases are between layers, so there are (num_layers - 1) sets
    for (int i = 0; i < nn->num_layers - 1; i++) {
        // Free the memory allocated for weights between layer i and i+1
        free(nn->weights[i]);
        // Free the memory allocated for biases of layer i+1
        free(nn->biases[i]);
    }

    // Loop through all layers
    for (int i = 0; i < nn->num_layers; i++) {
        // Free the memory allocated for activation values of layer i
        free(nn->activations[i]);
        // Free the memory allocated for z-values (weighted sums before activation) of layer i
        free(nn->z_values[i]);
    }

    // Free the arrays of pointers for weights, biases, activations, and z_values
    free(nn->weights);
    free(nn->biases);
    free(nn->activations);
    free(nn->z_values);

    // Note: The NeuralNetwork structure itself is not freed here, 
    // as it might have been allocated on the stack. If it was dynamically
    // allocated, the caller would need to free it separately.
}

// Function to perform forward propagation through the neural network
// Parameters:
//   nn: pointer to the NeuralNetwork structure
//   input: pointer to the input data array
void forward_propagation(NeuralNetwork* nn, double* input) {
    // Copy the input data to the activations of the first layer (input layer)
    // This uses memcpy to efficiently copy 'layer_sizes[0]' doubles
    memcpy(nn->activations[0], input, nn->layer_sizes[0] * sizeof(double));

    // Loop through each layer, starting from the second layer (index 1)
    for (int i = 1; i < nn->num_layers; i++) {
        // For each neuron in the current layer
        for (int j = 0; j < nn->layer_sizes[i]; j++) {
            // Initialize the sum with the bias of the current neuron
            // Note: biases are indexed with i-1 because there are no biases for the input layer
            double sum = nn->biases[i-1][j];
            
            // For each neuron in the previous layer
            for (int k = 0; k < nn->layer_sizes[i-1]; k++) {
                // Add the product of the activation of the k-th neuron in the previous layer
                // and the weight connecting it to the j-th neuron in the current layer
                // The index for weights is calculated as k * nn->layer_sizes[i] + j
                // This flattens the 2D weight matrix into a 1D array
                sum += nn->activations[i-1][k] * nn->weights[i-1][k * nn->layer_sizes[i] + j];
            }
            
            // Store the computed sum (before activation) in z_values
            nn->z_values[i][j] = sum;
            
            // Apply the sigmoid activation function to the sum and store the result
            // in the activations array for the current layer
            nn->activations[i][j] = sigmoid(sum);
        }
    }
}

// Function to perform backward propagation through the neural network
// Parameters:
//   nn: pointer to the NeuralNetwork structure
//   target: pointer to the target (expected) output array
//   learning_rate: the learning rate for weight updates
void backward_propagation(NeuralNetwork* nn, double* target, double learning_rate) {
    // Get the index of the output layer
    int output_layer = nn->num_layers - 1;
    
    // Allocate memory for the delta (error term) of the output layer
    double* delta = safe_malloc(nn->layer_sizes[output_layer] * sizeof(double));
    
    // Calculate the delta for the output layer
    for (int i = 0; i < nn->layer_sizes[output_layer]; i++) {
        // Compute the error: difference between actual and target output
        double error = nn->activations[output_layer][i] - target[i];
        // Calculate delta: error multiplied by derivative of activation function
        delta[i] = error * sigmoid_derivative(nn->activations[output_layer][i]);
    }

    // Propagate the error backwards through the network
    for (int l = output_layer - 1; l >= 0; l--) {
        // Allocate memory for the delta of the previous layer
        double* prev_delta = safe_malloc(nn->layer_sizes[l] * sizeof(double));
        
        // Calculate the delta for the current layer
        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            double error = 0.0;
            // Sum the weighted deltas from the next layer
            for (int j = 0; j < nn->layer_sizes[l+1]; j++) {
                error += delta[j] * nn->weights[l][i * nn->layer_sizes[l+1] + j];
            }
            // Calculate delta: error multiplied by derivative of activation function
            prev_delta[i] = error * sigmoid_derivative(nn->activations[l][i]);
        }

        // Update weights between current layer and next layer
        for (int i = 0; i < nn->layer_sizes[l]; i++) {
            for (int j = 0; j < nn->layer_sizes[l+1]; j++) {
                // Update rule: new_weight = old_weight - learning_rate * input_activation * delta
                nn->weights[l][i * nn->layer_sizes[l+1] + j] -= learning_rate * nn->activations[l][i] * delta[j];
            }
        }
        
        // Update biases of the next layer
        for (int i = 0; i < nn->layer_sizes[l+1]; i++) {
            // Update rule: new_bias = old_bias - learning_rate * delta
            nn->biases[l][i] -= learning_rate * delta[i];
        }

        // Free the memory for the current delta and move to the previous layer
        free(delta);
        delta = prev_delta;
    }

    // Free the memory for the last delta array
    free(delta);
}

// Function to draw a line on a 28x28 image
// Parameters:
//   image: pointer to the image array (assumed to be 28x28 = 784 elements)
//   m: slope of the line
//   b: y-intercept of the line
//   thickness: thickness of the line in pixels
void draw_line(double* image, double m, double b, int thickness) {
    // Iterate over each column of the image
    for (int x = 0; x < 28; x++) {
        // Calculate the y-coordinate for this x using the line equation y = mx + b
        // round() is used to get the nearest integer y-coordinate
        int y = round(m * x + b);
        
        // Iterate over a square region around the calculated point
        // This square's size is determined by the thickness
        for (int dx = -thickness/2; dx <= thickness/2; dx++) {
            for (int dy = -thickness/2; dy <= thickness/2; dy++) {
                // Calculate the actual pixel coordinates
                int px = x + dx;
                int py = y + dy;
                
                // Check if the pixel is within the image bounds
                if (px >= 0 && px < 28 && py >= 0 && py < 28) {
                    // If within bounds, set the pixel to 1.0 (white)
                    // The index is calculated as py * 28 + px to flatten the 2D coordinates
                    image[py * 28 + px] = 1.0;
                }
            }
        }
    }
}

// Function to generate a random image (either a cross or a circle)
// Parameters:
//   image: pointer to the image array (assumed to be 28x28 = 784 elements)
//   label: pointer to store the label of the generated image (0 for cross, 1 for circle)
void generate_image(double* image, int* label) {
    // Initialize the image array to all zeros (black background)
    memset(image, 0, INPUT_SIZE * sizeof(double));

    // Randomly determine the thickness of the lines (either 1 or 2 pixels)
    int thickness = 1 + rand() % 3;
    
    // Randomly choose between generating a cross (0) or a circle (1)
    *label = rand() % 2;
    
    if (*label == 0) {  // Generate a cross
        
        // Choose a random intersection point for the cross
        // The range is limited to avoid the cross being too close to the edges
        int ix = 4 + rand() % 20;  // x-coordinate of intersection (between 4 and 23)
        int iy = 4 + rand() % 20;  // y-coordinate of intersection (between 4 and 23)
        
        // Generate random slopes for the two lines
        // The slopes are between -1 and 1
        double m1 = (double)(rand() % 200 - 100) / 100;
        double m2;
        
        // Ensure the two slopes are sufficiently different to avoid near-parallel lines
        do {
            m2 = (double)(rand() % 200 - 100) / 100;
        } while (fabs(m2 - m1) < 0.1);

        // Calculate the y-intercepts for the two lines
        // Using the point-slope form: y - y1 = m(x - x1) => y = mx - mx1 + y1 => b = -mx1 + y1
        double b1 = iy - m1 * ix;
        double b2 = iy - m2 * ix;

        // Draw the two lines to form the cross
        draw_line(image, m1, b1, thickness);
        draw_line(image, m2, b2, thickness);
    } else {  // Generate a circle
        // Randomly determine the radius of the circle (between 5 and 12 pixels)
        int radius = 5 + rand() % 8;
        
        // Choose a random center point for the circle
        int center_x = rand() % 28;
        int center_y = rand() % 28;
        
        // Iterate over all pixels in the image
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                // Calculate the distance from the current pixel to the center of the circle
                double distance = sqrt(pow(x - center_x, 2) + pow(y - center_y, 2));
                
                // If the distance is within the thickness range of the radius, color the pixel white
                if (fabs(distance - radius) < thickness / 2.0) {
                    image[y * 28 + x] = 1.0;
                }
            }
        }
    }
}

// Function to print a visual representation of the image to the console
// Parameter:
//   image: pointer to the image array (assumed to be 28x28 = 784 elements)
void print_image(double* image) {
    // Iterate over each row of the image
    for (int i = 0; i < 28; i++) {
        // Iterate over each column of the image
        for (int j = 0; j < 28; j++) {
            // Decide which character to print based on the pixel value
            // If the pixel value is greater than 0.5, print '#' (representing a dark pixel)
            // Otherwise, print '.' (representing a light pixel)
            // The index i * 28 + j is used to access the correct element in the 1D array
            printf("%c", image[i * 28 + j] > 0.5 ? '#' : '.');
        }
        // After each row, print a newline character to move to the next line
        printf("\n");
    }
}

int main() {
    // Seed the random number generator with the current time
    // This ensures different random numbers each time the program is run
    srand(time(NULL));

    // Define the network architecture
    // This creates a network with 5 layers:
    // Input layer: INPUT_SIZE neurons (784 for 28x28 image)
    // First hidden layer: 64 neurons
    // Second hidden layer: 20 neurons
    // Third hidden layer: 20 neurons
    // Output layer: 2 neurons (one for each class: cross or circle)
    int num_layers = 5;
    int layer_sizes[] = {INPUT_SIZE, 64, 20, 20, 2};

    // Declare a NeuralNetwork structure
    NeuralNetwork nn;
    // Initialize the network with the defined architecture
    init_network(&nn, num_layers, layer_sizes);

    // Set the initial learning rate
    double learning_rate = INITIAL_LEARNING_RATE;

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        int correct = 0;  // Counter for correct predictions
        double total_error = 0.0;  // Accumulator for total error

        // Process each training example
        for (int i = 0; i < NUM_TRAINING_EXAMPLES; i++) {
            double input[INPUT_SIZE];
            int label;
            // Generate a random image (cross or circle) and its label
            generate_image(input, &label);

            // Perform forward propagation with the input
            forward_propagation(&nn, input);

            // Create the target output
            // [1,0] for cross (label 0), [0,1] for circle (label 1)
            double target[2] = {0, 0};
            target[label] = 1;

            // Calculate the error for this example
            double example_error = 0.0;
            for (int j = 0; j < 2; j++) {
                double diff = nn.activations[num_layers-1][j] - target[j];
                example_error += diff * diff;  // Sum of squared errors
            }
            total_error += example_error;

            // Perform backward propagation to update weights and biases
            backward_propagation(&nn, target, learning_rate);

            // Determine the network's prediction
            int predicted = nn.activations[num_layers-1][0] > nn.activations[num_layers-1][1] ? 0 : 1;
            // Increment correct count if prediction matches label
            if (predicted == label) correct++;
        }

        // Calculate average error for this epoch
        double avg_error = total_error / NUM_TRAINING_EXAMPLES;

        // Adjust the learning rate based on the epoch and average error
        learning_rate = adjust_learning_rate(learning_rate, epoch, avg_error);

        // Print the results for this epoch
        printf("Epoch %d, Accuracy: %.2f%%, Avg Error: %.4f, Learning Rate: %.4f\n", 
               epoch, (float)correct / NUM_TRAINING_EXAMPLES * 100, avg_error, learning_rate);
    }

    // Testing phase
    printf("\n\nTesting the neural network:\n\n");
    int correct = 0;
    // Test on 30 new examples
    for (int i = 0; i < 30; i++) {
        double input[INPUT_SIZE];
        int label;
        // Generate a new test image
        generate_image(input, &label);

        // Perform forward propagation to get the network's prediction
        forward_propagation(&nn, input);

        // Determine the prediction
        int predicted = nn.activations[num_layers-1][0] > nn.activations[num_layers-1][1] ? 0 : 1;

        // Increment correct count if prediction matches label
        if (predicted == label) correct++;

        // Print the actual and predicted labels
        printf("Actual: %s, Predicted: %s\n", 
               label == 0 ? "Cross" : "Circle", 
               predicted == 0 ? "Cross" : "Circle");
        // Print a visual representation of the image
        print_image(input);
        printf("\n");
    }

    // Print the overall test accuracy
    printf("Test Accuracy: %.2f%%\n", (float)correct / 30 * 100);

    // Free the memory allocated for the neural network
    free_network(&nn);

    return 0;
}
