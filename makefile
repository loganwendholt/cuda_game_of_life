life_serial: life_serial.c
	gcc life_serial.c -lGL -lGLU -lglut -o life_serial

life_parallel: life_parallel.cu
	nvcc life_parallel.cu -lGL -lGLU -lglut -o life_parallel
