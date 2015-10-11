life_serial: life_serial.c
	gcc life_serial.c -lGL -lGLU -lglut -o life_serial

life_parallel: life_parallel.cu
	nvcc life_parallel.cu -lGL -lGLU -lglut -o life_parallel

life_parallel_async: life_parallel_async.cu
	nvcc life_parallel_async.cu -lGL -lGLU -lglut -o life_parallel_async

life_parallel_streaming: life_parallel_streaming.cu
	nvcc life_parallel_streaming.cu -lGL -lGLU -lglut -o life_parallel_streaming

all:  life_serial life_parallel life_parallel_async life_parallel_streaming
