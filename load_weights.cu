FILE * ouvrirWb(char * nom){
  FILE * f = NULL;
  f = fopen(nom, "wb");
  assert(f);
  return f;
}

FILE * ouvrirRb(char * nom){
    FILE * f = NULL;
    f = fopen(nom, "rb");
    assert(f);
    return f;
}

void load_weights_from_saved(convNet * cnn, char * file_weights){
    FILE * f = NULL;
    int nb_composante = 3;
    float len_conv, len_FC;
    float * descripteur_conv, * descripteur_FC;
    f = ouvrirRb(argv[1]);
    printf("Lecture file\n");
    fread(&len_conv, sizeof(len_conv), 1, f);
    fread(&len_FC, sizeof(len_FC), 1, f);
    descripteur_conv = malloc(len_conv * sizeof *descripteur_conv);
    descripteur_FC = malloc(len_FC * sizeof *descripteur_FC);
    for(int i=0 ; i<(int)len_conv ; ++i){
        fread(&(descripteur_conv[i]), sizeof(*descripteur_conv), 1, f);
        printf("%f   %d\n", descripteur_conv[i], i);
    }
    for(int i=0 ; i<(int)len_FC ; ++i){
        fread(&(descripteur_FC[i]), sizeof(*descripteur_FC), 1, f);
        printf("%f   %d\n", descripteur_FC[i], i);
    }
    for(int i=0 ; i<(int)len_conv ; ++i){
        
    }
    /*
    printf("len_conv = %f    len_FC = %f\n", len_conv, len_FC);
    for(int i=0 ; i<35146 ; ++i){
        fread(&val, sizeof(val), 1, f);
        printf("%f   %d\n", val, i);
    }
    */
    printf("\n");
    fclose(f);
    return 0;
}