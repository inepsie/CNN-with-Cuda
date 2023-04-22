/*Le modèle ne demande qu’un prétraitement spécifique qui consiste à soustraire la valeur
RGB moyenne, calculée sur l’ensemble d’apprentissage, de chaque pixel. Source : https://datascientest.com/quest-ce-que-le-modele-vgg ???
*/
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<assert.h>
#include<float.h>

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
//#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
//#include <helper_cuda.h>         // helper functions for CUDA error check

struct conv_layer{
    //On stock tout les filtres d'une couche dans un vecteur 3D
    int h_input;
    int w_input;
    int depth_input;
    int nb_filters;
    int filter_size;
    int filter_depth;
    int len_w;//Taille de w et d_w = nb_filters * filter_size * depth
    float * w;//Poids des filtres en CPU
    float * d_w;//Poids des filtres en GPU
    int h_output;//Hauteur de l'output
    int w_output;//Largeur de l'output
    int len_output;
    float * output;//Stockage GPU de l'output de cette couche
    };
typedef struct conv_layer conv_layer;

struct FC_layer{
    //On stock les poids des neurones de la couche dans une vecteur 2D
    int len_w;//Taille de w et d_w = nb_neurones * nb_param
    float * w;
    float * d_w;
    float * output;//Stockage GPU de la sortie de cette couche. Taille output = nb_neurones
    int nb_neurones;//Hauteur
    int nb_param;//Largeur
};
typedef struct FC_layer FC_layer;

struct pool_layer{
    int w_input;
    int h_input;
    int w_output;
    int h_output;
    int depth_output;
    int len_output;
    float * output;
};
typedef struct pool_layer pool_layer;


struct convNet{
    //Données relatives au premier input du réseau, c'est à dire l'image. Elle est stockée avec le padding pour correspondre au format des vecteurs "output"
    //contenus dans les couches du réseau.
    float * input;
    float * input_d;
    int h_input;
    int w_input;
    int depth_input;
    //Données relatives aux couches de convolutions:
    int nb_conv_layers;
    conv_layer * vect_conv;
    //Données relatives aux couches de pooling;
    int nb_pooling_layers;
    pool_layer * vect_pool;
    //Données relatives aux couches Fully-Connected:
    int nb_FC_layers;
    FC_layer * vect_FC;
    //Premiers inputs des couches Fully-connected, dernier output des couches convolutions/pooling auquel on enlève le padding:
    int len_transi_conv_FC;
    float * transi_conv_FC;
    //Données relatives à la couche Softmax:
    int len_softmax;
    float * softmax;
};
typedef struct convNet convNet;