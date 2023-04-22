#define PRINT 0

#include "struct.h"
//FONCTIONS GPU :
//_____________________________
//Execution d'une couche de Convolution:
__global__ void convolution_01(float *inputData, int width, int height, float *filters, int filter_size, int filter_depth, int padding, int neg_padding, float *outputData);
__global__ void convolution_02(float *inputData, int len_out, int width, int height, float *filters, int filter_width, int filter_depth, int pos_padding, int neg_padding, float *outputData);
//Execution d'une couche de Max-Pooling:
__global__ void maxPooling(float * inputData, int width, int height, int padding_size, float * outputData, int lenData);
//Fait le lien entre la dernière couche de convolution/pooling et la première couche Fully-Connected en enlevant le padding:
__global__ void remove_padding(float * inputData, int width, int height, int padding_size, int len_out, float * outputData);
//Execution d'une couche Fully-Connected:
__global__ void fullyConnected_02(float * inputData, float * w, int nb_param, int nb_neurones, float * outputData);
//Mise à zéro du vecteur:
__global__ void init_zero(float * vect, int dimension);
//Fonction d'activation ReLU:
__global__ void relu(float * inputData, int len_output);
//Fonction softmax:
__global__ void softmax(float * inputData, int len_output, float * outputData);



//FONCTIONS CPU :
//_____________________________
//Initialisation des paramètres du réseau:
__host__ void init_param_cnn(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC, int len_descr_FC, int filter_size, int h_input, int w_input);
//Allocation mémoire CPU et GPU du réseau:
__host__ void alloc_cnn(convNet * cnn);
//Initialisation random des poids du réseau sur [0 - interval/2 ; interval/2]:
__host__ void init_weights_cnn(convNet * cnn, float interval);
/*Pour chaque couche on alloue la mémoire pour stocker la sortie de cette couche dans un vecteur. On met toutes les valeurs de celui-ci à zéro
et on ajoute des bords (ces deux choses pour gérer le zéro-padding):     */
__host__ void alloc_init_outputs(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC, int h_input, int w_input);
//Chargement de l'image en tant que premier input, ajout du padding;
__host__ void load_image(convNet * cnn, float * image, int height, int width, int depth, int filter_size);
//Execution du réseau de bout en bout :
__host__ void forward(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC);
//Print des infos du réseau
__host__ void print_w(convNet * cnn, int ind);
//Print couche de pooling
__host__ void print_pool(convNet * cnn, int ind);
