

//On prend la banque de filtres de convolution de taille impaire "filter_size" qu'on applique avec un pas "stride" pour former une sortie
//Dimension du filtre = filter_size * filter_size * 3 (3 pour RGB)
//En utilisant un "zero padding" l'output a les memes dimensions que l'input, sinon height = height - (filter_size - 1)


//Utilisé pour mettre les vecteurs output à zéro dans les couches de convolutions
__global__ void init_zero(float * data, int dimension){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<dimension){
        data[i] = 0.0;
    }
}


//Chaque bloc effectue un filtre 2d de taille 3*3
__global__ void convolution_01(float *inputData, int width, int height, float *filters, int filter_size, int filter_depth, int pos_padding, int neg_padding, float *outputData) {
    //Un block par filtre, un thread pas tranche de filtre
    //On traite un segment 2D de filtre c a d 3x3
    //Ensuite un thread élu fait la somme et insère dans output
    //(Aucune utilisation de la shared possible ? --> Problème d'accès a la globale mem ?)

    __shared__ float tmp[512]; // extern + pas de taille spécifiée

    tmp[threadIdx.x] = 0.0;

    int centre =    blockIdx.y * gridDim.x +
                    blockIdx.x +
                    (gridDim.x+2*(filter_size/2)) * (filter_size/2) + filter_size/2 + 2*(filter_size/2) * blockIdx.y;
    int centre_img = centre + threadIdx.x * (gridDim.x+2*(filter_size/2)) * (gridDim.y+2*(filter_size/2));

    int ind_filter = blockIdx.z * filter_size * filter_size * blockDim.x + threadIdx.x * filter_size * filter_size;//Partie constante en shared ?
    
    for(int j=neg_padding, j_filter=0 ; j<=pos_padding ; ++j, ++j_filter){
        for(int i=neg_padding, i_filter=0 ; i<=pos_padding ; ++i, ++i_filter){
            tmp[threadIdx.x] += inputData[centre_img + i + j * (gridDim.x + 2 * (filter_size/2))] * filters[ind_filter + i_filter + j_filter * filter_size];
            //tmp[threadIdx.x] += filters[ind_filter + i_filter + j_filter * filter_size];
            //tmp[threadIdx.x] += inputData[centre_img + i + j * (gridDim.x + 2 * (filter_size/2))];
            //tmp[threadIdx.x] += blockIdx.z;
        }
    }
    
    __syncthreads();

    if(threadIdx.x == 0){
        int centre_out = centre + blockIdx.z * (gridDim.x+2*(filter_size/2)) * (gridDim.y+2*(filter_size/2));
        float res = 0.0;
        //int n = blockIdx.z * blockDim.x * blockDim.y + blockIdx.y * blockDim.x + blockIdx.x + width + (3 + blockIdx.y * 2) * padding;
        for(int j=0 ; j<filter_depth ; ++j){
            res += tmp[j];
        }
        outputData[centre_out] = res;
    }
}

__global__ void convolution_02(float *inputData, int len_out, int width, int height, float *filters, int filter_width, int filter_depth, int pos_padding, int neg_padding, float *outputData) {
    int ind_thread = blockIdx.x * blockDim.x + threadIdx.x;
    if(ind_thread<len_out){
        int ind_filter = ind_thread / (width * height); //Calcul de l'indice du filtre utilisé par le thread
        int y = (ind_thread - (ind_filter * width * height)) / width; //Calcul de la coordonnée en y de la donnée de sortie produit par le thread
        int x = (ind_thread - (ind_filter * width * height)) % width; //Calcul de la coordonnée en x de la donnée de sortie produit par le thread
        float res = 0.0;
        int center_img =    y * width +
                            x +
                            (width + 2*pos_padding) * pos_padding + pos_padding + 2*pos_padding * y; //Ajout de l'offset produit par le padding
        int size_w_p = (width+2*pos_padding) * (height+2*pos_padding); //"size with padding" : Calcul de la taille de l'image (width*height) avec le padding.
        int filter_size = filter_width * filter_width * filter_depth; //Taille du filtre 3-dimensionel
        int ind_out = center_img + ind_filter * (width+2*pos_padding) * (height+2*pos_padding);

        for(int depth=0 ; depth<filter_depth ; ++depth){
            for(int j=neg_padding, j_filter=0 ; j<=pos_padding ; ++j, ++j_filter){
                for(int i=neg_padding, i_filter=0 ; i<=pos_padding ; ++i, ++i_filter){
                    res +=  inputData[center_img + depth*size_w_p + j*(width+2*pos_padding) + i] *  // Valeur de l'input multiplié par
                            filters[ind_filter*filter_size + depth*(filter_width*filter_width) + j_filter*filter_width + i_filter];                  // le poid du filtre associé
                    //res +=  inputData[center_img + depth*size_w_p + j*(width+2*pos_padding) + i];
                    //res +=  filters[ind_filter*filter_size + depth*(filter_width*filter_width) + j_filter*filter_width + i_filter];
                }
            }
        }
        outputData[ind_out] = res;
    }
}

__global__ void maxPooling(float * inputData, int width, int height, int padding_size, float * outputData, int lenData){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<lenData){
        int z = i / (width * height);
        int y = (i - (z * width * height)) / width;
        int x = (i - (z * width * height)) % width;
        int ind_out =   z * (width+2*padding_size) * (height+2*padding_size) +                  // Z
                        y * width +                                                             // Y
                        x +                                                                     // X
                        (width+2*padding_size) * padding_size + padding_size + 2*padding_size*y;// Offset

        int ind_in =    (z) * (2*width+2*padding_size) * (2*height+2*padding_size) +                // Z
                        (2*y) * 2*width +                                                             // Y
                        (2*x) +                                                                       // X
                        (2*width+2*padding_size) * padding_size + padding_size + 2*padding_size*(2*y);// Offset

        float max = inputData[ind_in];

        for(int n=0 ; n<2 ; ++n){
            for(int m=0 ; m<2 ; ++m){
                if(inputData[ind_in + m + n*(2*width+2*(padding_size))]>max){
                    max = inputData[ind_in + m + n*((2*width)+2*(padding_size))];
                }
            }
        }
        outputData[ind_out] = max;
    }
}

__global__ void remove_padding(float * inputData, int width, int height, int padding_size, int len_out, float * outputData){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<len_out){
        int z = i / (width * height);
        int y = (i - (z * width * height)) / width;
        int x = (i - (z * width * height)) % width;
        int ind_in =   z * (width+2*padding_size) * (height+2*padding_size) +                       // Z
                            y * width +                                                             // Y
                            x +                                                                     // X
                            (width+2*padding_size) * padding_size + padding_size + 2*padding_size*y;// Offset
        outputData[i] = inputData[ind_in];
    }
}

__global__ void fullyConnected(float * inputData, float * w, int nb_param, int nb_neurones, float * outputData){
    
    //Inserez les multiplications dans un vecteur de shared memory
    extern __shared__ float tmp[]; // extern + pas de taille spécifiée
    __shared__ float input[1024]; //Partie de l'input correspondant au bloc

    //Mettre la partie de l'input correspondant au bloc en shared_memory
    if(threadIdx.x==0){
        for(int j=0 ; j<1024 ; ++j){
            input[j] = inputData[blockIdx.x * blockDim.x + j];
        }
    }
    //__syncthreads(); // barrière de synchro

    //Résultat d'un neurone initialisé à zéro:
    if(blockIdx.x == 0 && threadIdx.x == 0) outputData[blockIdx.y] = 0;
    __syncthreads(); // barrière de synchro

    //Chaque thread effectue la mutliplication d'un paramètre avec le poid du neurone qui lui est associé:
    tmp[threadIdx.x] = input[threadIdx.x] * w[blockIdx.y * nb_param + blockIdx.x * 1024 + threadIdx.x];
    __syncthreads(); // barrière de synchro
    
    //Un élu par block pour faire la somme partielle
    if(threadIdx.x==0){
        float sum = 0;
        for(int j=0 ; j<blockDim.x ; ++j){
            sum += tmp[j];
        }
        //puis on insere la somme partielle dans S via atomicAdd
        //garantissant un accès s^ur en mode concurrentiel
        atomicAdd(&(outputData[blockIdx.y]), sum);
    }
}

//Un thread par neurone, donc par output:
__global__ void fullyConnected_02(float * inputData, float * w, int nb_param, int nb_neurones, float * outputData){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<nb_neurones){
        float tmp = 0.0;
        int ind_w = i * nb_param;
        for(int j=0 ; j<nb_param ; ++j){
            tmp += inputData[j] * w[ind_w + j];
        }
        outputData[i] = tmp;
    }
}

__global__ void relu(float * inputData, int len_output){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<len_output){
        inputData[i] = fmaxf(0, inputData[i]);
    }
}

__global__ void softmax(float * inputData, int len_output, float * outputData){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<len_output){
        float tmp = 0.0;
        for(int j=0 ; j<len_output ; ++j){
            tmp += exp(inputData[j]);
        }
        outputData[i] = exp(inputData[i]) / tmp;
    }
}


