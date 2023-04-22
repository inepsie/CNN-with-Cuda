#include "functions.h"

__host__ void init_param_cnn(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC, int len_descr_FC, int filter_size, int h_input, int w_input){
    //Initialisation des variables de paramètres du réseau

    //Calcul du nombre de chaque type de couche + allocation des vecteurs contenant ces couches : 
    cnn->nb_conv_layers = 0;
    cnn->nb_pooling_layers = 0;
    for(int i=0 ; i<len_descr_conv ; ++i){
        if(descripteur_conv[i]>0) ++cnn->nb_conv_layers;
        else if(descripteur_conv[i]==0) ++cnn->nb_pooling_layers;
    }
    cnn->nb_FC_layers = len_descr_FC;
    cnn->vect_conv = (conv_layer*) malloc(cnn->nb_conv_layers * sizeof(*(cnn->vect_conv)));
    cnn->vect_FC = (FC_layer*) malloc(cnn->nb_FC_layers * sizeof(*(cnn->vect_FC)));
    cnn->vect_pool = (pool_layer*) malloc(cnn->nb_pooling_layers * sizeof(*(cnn->vect_pool)));
    
    //Initialisation des paramètres des couches de convolutions (nombre de filtres, profondeur):
    cnn->vect_conv[0].filter_depth = 3;// 3 pour RGB
    cnn->vect_conv[0].nb_filters = descripteur_conv[0];
    cnn->vect_conv[0].filter_size = filter_size;//AJOUT DE LA POSSIBILITE D'AVOIR DIFFERENTES TAILLES DE FILTRES ?
    cnn->vect_conv[0].len_w = cnn->vect_conv[0].nb_filters * cnn->vect_conv[0].filter_depth * cnn->vect_conv[0].filter_size * cnn->vect_conv[0].filter_size;
    for(int i=1, conv=1; i<len_descr_conv ; ++i){
        if(descripteur_conv[i]>0){
            cnn->vect_conv[conv].nb_filters = descripteur_conv[i];
            cnn->vect_conv[conv].filter_depth = cnn->vect_conv[conv-1].nb_filters;
            cnn->vect_conv[conv].filter_size = filter_size;
            cnn->vect_conv[conv].len_w = cnn->vect_conv[conv].nb_filters * cnn->vect_conv[conv].filter_depth * cnn->vect_conv[conv].filter_size * cnn->vect_conv[conv].filter_size;
            ++conv;
        }
    }
    //Initialisation des paramètres des couches FC (nombre de neurones, nombre de paramètres des neurones):
    cnn->vect_FC[0].nb_neurones = descripteur_FC[0];
    cnn->vect_FC[0].nb_param = (h_input / pow(2,cnn->nb_pooling_layers)) * (w_input / pow(2,cnn->nb_pooling_layers));
    cnn->vect_FC[0].len_w = cnn->vect_FC[0].nb_neurones * cnn->vect_FC[0].nb_param;
    for(int i=1; i<cnn->nb_FC_layers ; ++i){
        cnn->vect_FC[i].nb_param = cnn->vect_FC[i-1].nb_neurones;
        cnn->vect_FC[i].nb_neurones = descripteur_FC[i];
        cnn->vect_FC[i].len_w = cnn->vect_FC[i].nb_neurones * cnn->vect_FC[i].nb_param;
    }
    //Initialisation des paramètres de la couche Softmax:
    cnn->len_softmax = cnn->vect_FC[cnn->nb_FC_layers-1].nb_neurones;
}

//Allocation mémoire CPU et GPU du réseau.
void alloc_cnn(convNet * cnn){
    for(int i=0 ; i<cnn->nb_conv_layers ; ++i){
        //Allocation des filtres en CPU:
        cnn->vect_conv[i].w = (float*) malloc(cnn->vect_conv[i].len_w * sizeof(*(cnn->vect_conv[i].w)));
        //Allocation des filtres en GPU:
        cudaMalloc(&cnn->vect_conv[i].d_w, cnn->vect_conv[i].len_w * sizeof(*(cnn->vect_conv[i].d_w)));
    }
    for(int i=0 ; i<cnn->nb_FC_layers ; ++i){
        //Allocation des couches Fully-Connected en CPU:
        cnn->vect_FC[i].w = (float*) malloc(cnn->vect_FC[i].nb_neurones * cnn->vect_FC[i].nb_param * sizeof(*(cnn->vect_FC[i].w)));
        //Allocation des couches Fully-Connected en GPU:
        cudaMalloc(&cnn->vect_FC[i].d_w, cnn->vect_FC[i].nb_neurones * cnn->vect_FC[i].nb_param * sizeof(*(cnn->vect_FC[i].w)));
    }
}

//Initialisation random des poids du réseau sur [0 - interval/2 ; interval/2]
void init_weights_cnn(convNet * cnn, float interval){
    float x;
    if(PRINT){ printf("\n\n");}
    for(int i=0 ; i<cnn->nb_conv_layers ; ++i){
        if(PRINT){ printf("\n\n");}
        if(PRINT){ printf("vect_conv[%d].len_w = %d\n", i, cnn->vect_conv[i].len_w);}
        for(int ind=0 ; ind<cnn->vect_conv[i].len_w ; ++ind){
            x = (float)rand()/(float)(RAND_MAX/(interval)) - interval/2;//Remplacer par "glorot init" ?? ou autre 
            cnn->vect_conv[i].w[ind] = x;
            if(PRINT){ printf("%f ", x);}
        }
        cudaMemcpy(cnn->vect_conv[i].d_w, cnn->vect_conv[i].w, cnn->vect_conv[i].len_w * sizeof(*cnn->vect_conv[i].d_w), cudaMemcpyHostToDevice);
        free(cnn->vect_conv[i].w);
    }
    for(int i=0 ; i<cnn->nb_FC_layers ; ++i){
        if(PRINT){ printf("\n");}
        for(int ind=0 ; ind<cnn->vect_FC[i].len_w ; ++ind){
            x = (float)rand()/(float)(RAND_MAX/(interval)) - interval/2;//Remplacer par "glorot init" ?? ou autre 
            cnn->vect_FC[i].w[ind] = x;
            if(PRINT){ printf("%f ", x);}
        }
        cudaMemcpy(cnn->vect_FC[i].d_w, cnn->vect_FC[i].w, cnn->vect_FC[i].len_w * sizeof(*cnn->vect_FC[i].d_w), cudaMemcpyHostToDevice);
        free(cnn->vect_FC[i].w);
    }
}

void load_image(convNet * cnn, float * image, int height, int width, int depth, int filter_size){//FAIRE EN GPU
    int len = (height + 2*(filter_size/2)) * (width + 2*(filter_size/2)) * depth;
    cnn->h_input = height;
    cnn->w_input = width;
    cnn->depth_input = depth;
    cnn->input = (float*) malloc( len * depth * sizeof *(cnn->input));
    cudaMalloc(&(cnn->input_d), len * depth * sizeof *(cnn->input));
    memset(cnn->input, 2, len * sizeof *(cnn->input));
    cudaMemset(cnn->input_d, 0, len * sizeof *(cnn->input));
    for(int z=0 ; z<depth ; ++z){
        for(int y=0 ; y<height ; ++y){
            for(int x=0 ; x<width ; ++x){
                cnn->input[
                (width+2*(filter_size/2)) * (filter_size/2) + filter_size/2 + 2*(filter_size/2) * y +//Padding
                z * (width+2*(filter_size/2)) * (height+2*(filter_size/2)) +//Depth
                y * width +//Height
                x]
                = image[z * width * height + y * width + x];//Width
            }
        }
    }
    if(PRINT){ printf("\n\n");}
    for(int i=0 ; i<len ; ++i){
        if(PRINT){ printf("%f ", cnn->input[i]);}
    }
    cudaMemcpy(cnn->input_d, cnn->input, len * sizeof * (cnn->input_d), cudaMemcpyHostToDevice);
}

//Pour chaque couche on alloue la mémoire pour stocker la sortie de cette couche dans un vecteur. On met toutes les valeurs de celui-ci à zéro
//et on ajoute des bords (ces deux choses pour gérer le zéro-padding). 
void alloc_init_outputs(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC, int h_input, int w_input){
    int output_len = 0;
    if(PRINT){ printf("\nLEN DESCR CONV = %d\n", len_descr_conv);}
    for(int i=0, conv=0, pool=0 ; i<len_descr_conv ; ++i){
        if(descripteur_conv[i]==0){
            assert(conv-1>=0);
            cnn->vect_pool[pool].h_input = h_input;
            cnn->vect_pool[pool].w_input = w_input;
            h_input /= 2;
            w_input /= 2;
            cnn->vect_pool[pool].h_output = h_input+2*(cnn->vect_conv[conv-1].filter_size/2);
            cnn->vect_pool[pool].w_output = w_input+2*(cnn->vect_conv[conv-1].filter_size/2);
            cnn->vect_pool[pool].depth_output = cnn->vect_conv[conv-1].nb_filters;
            cnn->vect_pool[pool].len_output = cnn->vect_pool[pool].h_output * cnn->vect_pool[pool].w_output * cnn->vect_pool[pool].depth_output;
            cudaMalloc(&(cnn->vect_pool[pool].output), cnn->vect_pool[pool].len_output * sizeof * (cnn->vect_pool[pool].output));
            cudaMemset(cnn->vect_pool[pool].output, 0, cnn->vect_pool[pool].len_output * sizeof * (cnn->vect_pool[pool].output));
            ++pool;
            continue;
        }
        if(descripteur_conv[i]>0){
            cnn->vect_conv[conv].w_input = w_input;
            cnn->vect_conv[conv].h_input = h_input;
            cnn->vect_conv[conv].h_output = h_input+2*(cnn->vect_conv[conv].filter_size/2); //On ajoute l'espace pour zero-padding
            cnn->vect_conv[conv].w_output = w_input+2*(cnn->vect_conv[conv].filter_size/2); //On ajoute l'espace pour zero-padding
            output_len = cnn->vect_conv[conv].h_output * cnn->vect_conv[conv].w_output * cnn->vect_conv[conv].nb_filters;
            cnn->vect_conv[conv].len_output = output_len;
            cudaMalloc(&(cnn->vect_conv[conv].output), output_len * sizeof * cnn->vect_conv[conv].output);
            cudaDeviceSynchronize();
            cudaMemset(cnn->vect_conv[conv].output, 0, output_len * sizeof * cnn->vect_conv[conv].output);
            cudaDeviceSynchronize();
            ++conv;
        }
    }
    cnn->len_transi_conv_FC = h_input * w_input * cnn->vect_conv[cnn->nb_conv_layers-1].nb_filters;
    cudaMalloc(&(cnn->transi_conv_FC), cnn->len_transi_conv_FC * sizeof * (cnn->transi_conv_FC));

    for(int i=0 ; i<cnn->nb_FC_layers ; ++i){
        /*https://www.analyticsvidhya.com/blog/2020/10/what-is-the-convolutional-neural-network-architecture/
        Finally, we flatten all the 5 x 5 x 16 to a single layer of size 400 values an inputting them to a 
        feed-forward neural network of 120 neurons having a weight matrix of size [400,120] and a hidden layer 
        of 84 neurons connected by the 120 neurons with a weight matrix of [120,84] and these 84 neurons indeed 
        are connected to a 10 output neurons
        */
        cudaMalloc(&(cnn->vect_FC[i].output), cnn->vect_FC[i].nb_neurones * sizeof * cnn->vect_FC[i].output);
        cudaDeviceSynchronize();
    }

    cudaMalloc(&(cnn->softmax), cnn->len_softmax * sizeof * (cnn->softmax));
}

__host__ void forward(convNet * cnn, int * descripteur_conv, int len_descr_conv, int * descripteur_FC){
    int type = 0; // Indique le type de couche qu'on vient de traiter (0:conv, 1:pooling)
    int len_out = cnn->vect_conv[0].w_input * cnn->vect_conv[0].h_input * cnn->vect_conv[0].nb_filters;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (len_out + 1024 - 1) / 1024;
    convolution_02  <<<blocksPerGrid, threadsPerBlock, 0>>>
                    (cnn->input_d,
                    len_out,
                    cnn->vect_conv[0].w_input,
                    cnn->vect_conv[0].h_input,
                    cnn->vect_conv[0].d_w,
                    cnn->vect_conv[0].filter_size,
                    cnn->vect_conv[0].filter_depth,
                    cnn->vect_conv[0].filter_size/2,
                    -1 * cnn->vect_conv[0].filter_size/2,
                    cnn->vect_conv[0].output);
    if(PRINT){ print_w(cnn, 0);}
    relu<<<(cnn->vect_conv[0].len_output + 1024 - 1) / 1024, 1024>>>(cnn->vect_conv[0].output, cnn->vect_conv[0].len_output);
    if(PRINT){ print_w(cnn, 0);}
                     
    for(int i=1 , conv=1, pool=0; i<len_descr_conv ; ++i){
        if(descripteur_conv[i]>0 && type==0){//CONVOLUTION FROM CONVOLUTION OUTPUT
            if(PRINT){ printf("CONVOLUTION FROM CONVOLUTION OUTPUT\n");}
            len_out = cnn->vect_conv[conv].w_input * cnn->vect_conv[conv].h_input * cnn->vect_conv[conv].nb_filters;
            blocksPerGrid = (len_out + 1024 - 1) / 1024;
            convolution_02  <<<blocksPerGrid, threadsPerBlock, 0>>>(
                            cnn->vect_conv[conv-1].output,
                            len_out,
                            cnn->vect_conv[conv].w_input,
                            cnn->vect_conv[conv].h_input,
                            cnn->vect_conv[conv].d_w,
                            cnn->vect_conv[conv].filter_size,
                            cnn->vect_conv[conv].filter_depth,
                            cnn->vect_conv[conv].filter_size/2,
                            -1 * cnn->vect_conv[conv].filter_size/2,
                            cnn->vect_conv[conv].output);
            if(PRINT){ print_w(cnn, conv);}
            relu<<<(cnn->vect_conv[conv].len_output + 1024 - 1) / 1024, 1024>>>(cnn->vect_conv[conv].output, cnn->vect_conv[conv].len_output);
            if(PRINT){ print_w(cnn, conv);}
            ++conv;
            type=0;
            continue;
        }
        if(descripteur_conv[i]>0 && type==1){//CONVOLUTION FROM POOLING OUTPUT
            if(PRINT){ printf("CONVOLUTION FROM POOLING OUTPUT\n");}
            len_out = cnn->vect_conv[conv].w_input * cnn->vect_conv[conv].h_input * cnn->vect_conv[conv].nb_filters;
            blocksPerGrid = (len_out + 1024 - 1) / 1024;
            convolution_02  <<<blocksPerGrid, threadsPerBlock, 0>>>(
                            cnn->vect_conv[pool-1].output,
                            len_out,
                            cnn->vect_conv[conv].w_input,
                            cnn->vect_conv[conv].h_input,
                            cnn->vect_conv[conv].d_w,
                            cnn->vect_conv[conv].filter_size,
                            cnn->vect_conv[conv].filter_depth,
                            cnn->vect_conv[conv].filter_size/2,
                            -1 * cnn->vect_conv[conv].filter_size/2,
                            cnn->vect_conv[conv].output);
            if(PRINT){ print_w(cnn, conv);}
            relu<<<(cnn->vect_conv[conv].len_output + 1024 - 1) / 1024, 1024>>>(cnn->vect_conv[conv].output, cnn->vect_conv[conv].len_output);
            if(PRINT){ print_w(cnn, conv);}
            ++conv;
            type=0;
            continue;
        }
        if(descripteur_conv[i]==0){//POOLING FROM CONVOLUTION OUTPUT
            len_out = cnn->vect_conv[conv-1].w_input/2 * cnn->vect_conv[conv-1].h_input/2 * cnn->vect_conv[conv-1].nb_filters;
            blocksPerGrid = (len_out + 1024 - 1) / 1024;
            maxPooling  <<<blocksPerGrid, threadsPerBlock, 0>>>
                        (cnn->vect_conv[conv-1].output,
                        cnn->vect_pool[pool].w_input/2,
                        cnn->vect_pool[pool].h_input/2,
                        cnn->vect_conv[conv-1].filter_size/2, //// A changer : indexation sur la taille du filtre de la couche suivante
                        cnn->vect_pool[pool].output,
                        len_out);
            if(PRINT){ print_pool(cnn, pool);}
            ++pool;
            type=1;
            continue;
        }
    }
    
    if(type==0){
        remove_padding  <<<(cnn->len_transi_conv_FC + 1024 - 1) / 1024, 1024>>>
                        (cnn->vect_conv[cnn->nb_conv_layers-1].output,
                        cnn->vect_conv[cnn->nb_conv_layers-1].w_input,
                        cnn->vect_conv[cnn->nb_conv_layers-1].h_input,
                        (cnn->vect_conv[cnn->nb_conv_layers-1].filter_size)/2,
                        cnn->len_transi_conv_FC,
                        cnn->transi_conv_FC);
    }
    else{
        remove_padding  <<<(cnn->len_transi_conv_FC + 1024 - 1) / 1024, 1024>>>
                        (cnn->vect_pool[cnn->nb_pooling_layers-1].output,
                        cnn->vect_pool[cnn->nb_pooling_layers-1].w_input/2,
                        cnn->vect_pool[cnn->nb_pooling_layers-1].h_input/2,
                        (cnn->vect_conv[cnn->nb_conv_layers-1].filter_size)/2,
                        cnn->len_transi_conv_FC,
                        cnn->transi_conv_FC);
    }

    //Lancement de la première couche FC qui prend son entrée depuis "cnn->transi_conv_FC":
    fullyConnected_02  <<<(cnn->vect_FC[0].nb_neurones + 1024 - 1) / 1024, 1024>>>
                        (cnn->transi_conv_FC,
                        cnn->vect_FC[0].d_w,
                        cnn->vect_FC[0].nb_param,
                        cnn->vect_FC[0].nb_neurones,
                        cnn->vect_FC[0].output);
    relu<<<(cnn->vect_FC[0].nb_neurones + 1024 - 1) / 1024, 1024>>>(cnn->vect_FC[0].output, cnn->vect_FC[0].nb_neurones);

    //Lancement des couches FC qui prennent leur entrée sur l'output de la couche FC précédente:
    for(int i=1 ; i<cnn->nb_FC_layers ; ++i){
        fullyConnected_02   <<<(cnn->vect_FC[i].nb_neurones + 1024 - 1) / 1024, 1024>>>
                            (cnn->vect_FC[i-1].output,
                            cnn->vect_FC[i].d_w,
                            cnn->vect_FC[i].nb_param,
                            cnn->vect_FC[i].nb_neurones,
                            cnn->vect_FC[i].output);
        relu<<<(cnn->vect_FC[i].nb_neurones + 1024 - 1) / 1024, 1024>>>(cnn->vect_FC[i].output, cnn->vect_FC[i].nb_neurones);
    }

    softmax<<<( cnn->len_softmax + 1024 - 1) / 1024, 1024>>>(cnn->vect_FC[cnn->nb_FC_layers-1].output, cnn->len_softmax, cnn->softmax);
}