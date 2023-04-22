# Location of the CUDA Toolkit
CUDA_PATH?=/usr/local/cuda/

all:
	nvcc *.cu -lm -o cnn

clean: 
	rm -rf *.o core
	rm -f core
	rm -rf *~
	rm -rf cnn
run:
	./cnn
