#!/usr/bin/python

from subprocess import call

def setGameParameters(window_width, window_height, game_width, game_height, num_blocks, num_threads, iterations):
	f = open("life_config.h", "w")

	f.write("#define WINDOW_WIDTH "+str(window_width)+"    // width of window in pixels\n")
	f.write("#define WINDOW_HEIGHT "+str(window_height)+"   // height of window in pixels\n")
	f.write("#define GAME_WIDTH "+str(game_width)+"      // number of cells it takes to span the window horizontally\n")
	f.write("#define GAME_HEIGHT "+str(game_height)+"      // number of cells it takes to span the window vertically\n")
	f.write("\n")
	f.write("#define NUM_BLOCKS "+str(num_blocks)+"\n")
	f.write("#define NUM_THREADS "+str(num_threads)+"\n")
	
	f.write("#define ITERATIONS "+str(iterations)+"     // number of simulation steps per run\n")
	f.write("#define DEBUG_MODE          // turn on debug mode")
	return



window_width = 640
window_height = 480
game_width = 128
game_height = 96
iterations = 1000

blocksThreads = ([game_width*game_height/16, 16], 
					[game_width*game_height/32, 32], 
					[game_width*game_height/64, 64], 
					[game_width*game_height/128, 128],
					[game_width*game_height/192, 192],
					[game_width*game_height/256, 256],
					[game_width*game_height/512, 512])

serial_targets = ['life_serial']
parallel_targets = ['life_parallel', 'life_parallel_async', 'life_parallel_streaming']

for target in serial_targets:
	f = open(target+"_data.txt", "w")
	setGameParameters(window_width, window_height, game_width, game_height, game_height, game_width, iterations)
	call(["make",target])
	call(["./"+target], stdout=f)


for target in parallel_targets:
	for bt in blocksThreads:
		f = open(target+"_data"+str(bt[0])+"x"+str(bt[1])+".txt", "w")
		setGameParameters(window_width, window_height, game_width, game_height, bt[0], bt[1], iterations)
		call(["make",target])
		call(["./"+target], stdout=f)






