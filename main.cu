#include "functions.h"


static struct timeval ti;

__host__ void initTime(void){
    gettimeofday(&ti, (struct timezone*) 0);
}

__host__ double getTime(void){
    struct timeval t;
    double diff;
    gettimeofday(&t, (struct timezone*) 0);
    diff = (t.tv_sec - ti.tv_sec) * 1000000 + (t.tv_usec - ti.tv_usec);
    return diff/1000.;
}

__host__ void print_w(convNet * cnn, int ind){
    printf("\n\nCOUCHE NUMERO : %d\n", ind);
    printf("Len_w = %d\n", cnn->vect_conv[ind].len_w);
    printf("Nb_conv_layers=%d\n", cnn->nb_conv_layers);
    printf("LEN_OUTPUT = %d\n", cnn->vect_conv[ind].len_output);
    printf("w_input = %d\n", cnn->vect_conv[ind].w_input);
    printf("h_input = %d\n", cnn->vect_conv[ind].h_input);
    printf("w_output = %d\n", cnn->vect_conv[ind].w_output);
    printf("h_output = %d\n", cnn->vect_conv[ind].h_output);
    printf("Nb_filters = %d\n", cnn->vect_conv[ind].nb_filters);
    printf("filter_size = %d\n", cnn->vect_conv[ind].filter_size);
    printf("filter_depth = %d\n", cnn->vect_conv[ind].filter_depth);
    printf("filter_size / 2 = %d\n", cnn->vect_conv[ind].filter_size/2);
    float * copie = (float *) malloc(cnn->vect_conv[ind].len_output * sizeof(float));
    cudaMemcpy(copie, cnn->vect_conv[ind].output, cnn->vect_conv[ind].len_output * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nAFFICHAGE OUTPUT[%d]\n", ind);
    /*
    for(int plop=0 ; plop<cnn->vect_conv[ind].len_output ; ++plop){
        printf("%f  ", copie[plop]);
    }
    */
    
    for(int d=0 ; d<cnn->vect_conv[ind].nb_filters ; ++d){
        printf("\n\n");
        for(int j=0 ; j<cnn->vect_conv[ind].h_output ; ++j){
            printf("\n");
            for(int i=0 ; i<cnn->vect_conv[ind].w_output ; ++i){
                //printf("%d  ", d*cnn->vect_conv[ind].w_output*cnn->vect_conv[ind].h_output + j*cnn->vect_conv[ind].w_output + i);
                printf("%f  ", copie[d*cnn->vect_conv[ind].w_output*cnn->vect_conv[ind].h_output + j*cnn->vect_conv[ind].w_output + i]);
            }
        }
    }
    printf("\n");
    free(copie);
}

__host__ void print_pool(convNet * cnn, int ind){
    printf("\nPRINT POOL\n");
    printf("Indice pool = %d\n", ind);
    printf("h_out = %d,   w_out=%d\n", cnn->vect_pool[ind].h_output, cnn->vect_pool[ind].w_output);
    printf("len_out = %d\n", cnn->vect_pool[ind].len_output);
    float * copie = (float*) malloc(cnn->vect_pool[ind].len_output * sizeof(float));
    cudaMemcpy(copie, cnn->vect_pool[ind].output, cnn->vect_pool[ind].len_output * sizeof * copie, cudaMemcpyDeviceToHost);
    for(int depth=0 ; depth<cnn->vect_pool[ind].depth_output ; ++depth){
        printf("\n\n");
        for(int j=0 ; j<cnn->vect_pool[ind].h_output ; ++j){
            printf("\n");
            for(int i=0 ; i<cnn->vect_pool[ind].w_output ; ++i){
                printf("%f  ", copie[i + j*cnn->vect_pool[ind].w_output + depth*cnn->vect_pool[ind].w_output*cnn->vect_pool[ind].h_output]);
            }
        }
    }
    printf("\n");
}

__host__ void print_transi(convNet * cnn){
    float * copie = (float*) malloc(cnn->len_transi_conv_FC * sizeof(*copie));
    cudaMemcpy(copie, cnn->transi_conv_FC, cnn->len_transi_conv_FC * sizeof * copie, cudaMemcpyDeviceToHost);
    for(int i=0 ; i<cnn->len_transi_conv_FC ; ++i){
        printf("%f  ", copie[i]);
    }
    printf("\n");
}


__host__ void print_FC(convNet * cnn, int ind){
    float * copie = (float*) malloc(cnn->vect_FC[ind].nb_neurones * sizeof * copie);
    printf("Transi conv FC = %d\n", cnn->len_transi_conv_FC);
    printf("Len_w = %d\n", cnn->vect_FC[ind].len_w);
    printf("Nb_neurones = %d\n", cnn->vect_FC[ind].nb_neurones);
    printf("Nb_param = %d\n", cnn->vect_FC[ind].nb_param);
    cudaMemcpy(copie, cnn->vect_FC[ind].output, cnn->vect_FC[ind].nb_neurones * sizeof * copie, cudaMemcpyDeviceToHost);
    printf("\n");
    for(int i=0 ; i<cnn->vect_FC[ind].nb_neurones ; ++i){
        printf("%f  ", copie[i]);
    }
    printf("\n");
}

__host__ void print_softmax(convNet * cnn){
    float somme = 0.0;
    float x;
    float * copie = (float*) malloc(cnn->len_softmax * sizeof * copie);
    printf("Len_softmax = %d\n", cnn->len_softmax);
    cudaMemcpy(copie, cnn->softmax, cnn->len_softmax * sizeof * copie, cudaMemcpyDeviceToHost);
    for(int i=0 ; i<cnn->len_softmax ; ++i){
        x = copie[i];
        somme += x;
        printf("%E  ", x);
    }
    printf("\nSomme Softmax = %f\n", somme);
}

int main(void){
    srand(6);
    initTime();
    int width = 224;
    int height = 224;
    //int width = 32;
    //int height = 32;
    float * image = (float*) malloc(3 * width * height * sizeof(*image));
    //printf("\nAFFICHAGE VALEURS IMAGE\n");
    for(int i=0 ; i<3*width*height ; ++i){
        image[i] = (float)rand()/(float)(RAND_MAX/(255.0)) - (255.0 / 2);
        //image[i] = i;
        //printf("%f  ", image[i]);
    }
    
    convNet cnn;
    int filter_size = 3;
    float interval_weights = 0.07;
    /*Initialisation des paramètres du réseau
        Descripteur : description des couches du réseau. Chaque case du vecteur représente une couche :
            - Un entier N > 0 décrit une couche convolution à N filtres
            - Un entier N == 0 représente une couche de Pooling
    */
    int descripteur_conv[] = {6, 0, 12, 0, 12, 0};
    //int descripteur_conv[] = {10};
    //int descripteur_FC[] = {10};
    int descripteur_FC[] = {120, 60, 40, 10};
    int len_descr_conv = sizeof(descripteur_conv)/sizeof(*descripteur_conv);
    int len_descr_FC = sizeof(descripteur_FC)/sizeof(*descripteur_FC);
    

    init_param_cnn(&cnn, descripteur_conv, len_descr_conv, descripteur_FC, len_descr_FC, filter_size, height, width);
    alloc_cnn(&cnn);//Allocation mémoire du réseau
    init_weights_cnn(&cnn, interval_weights);//Initialisation random des poids du réseau [0 - interval/2 ; interval/2]
    alloc_init_outputs(&cnn, descripteur_conv, len_descr_conv, descripteur_FC, height, width);

    //void load_image(convNet * cnn, float * image, int height, int width, int depth, int filter_size){
    load_image(&cnn, image, height, width, 3, 3);
    forward(&cnn, descripteur_conv, len_descr_conv, descripteur_FC);
    //cudaDeviceSynchronize();
    printf("\nTime = %F\n", getTime());
    
    if(PRINT){
        printf("\n\nLEN CONV LAYERS = %d\n", len_descr_conv);
        printf("LEN FC01 LAYERS = %d\n", len_descr_FC);
        printf("\n\nNb_conv_layers=%d\n", cnn.nb_conv_layers);
        printf("Nb_pool_layers=%d\n", cnn.nb_pooling_layers);

        printf("\nTRANSI\n");
        printf("Len Transi = %d\n", cnn.len_transi_conv_FC);
        print_transi(&cnn);

        printf("\nFULLY CONNECT\n");
        for(int i=0 ; i<cnn.nb_FC_layers ; ++i){
            printf("FC numéro %d\n", i);
            print_FC(&cnn, i);
        }
    }
    print_softmax(&cnn);
       
}