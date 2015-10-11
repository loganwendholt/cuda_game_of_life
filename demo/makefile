life_serial: life_serial.c life_config.h
	gcc life_serial.c -lGL -lGLU -lglut -o life_serial

life_parallel: life_parallel.cu life_config.h
	nvcc life_parallel.cu -lGL -lGLU -lglut -o life_parallel

life_parallel_async: life_parallel_async.cu life_config.h
	nvcc life_parallel_async.cu -lGL -lGLU -lglut -o life_parallel_async

life_parallel_streaming: life_parallel_streaming.cu life_config.h
	nvcc life_parallel_streaming.cu -lGL -lGLU -lglut -o life_parallel_streaming

all:  life_serial life_parallel life_parallel_async life_parallel_streaming
