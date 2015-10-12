import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
import os
import re
from operator import add

#######################################################################
#  Function:  generate_chart
#
#  Author:    Logan Wendholt
#
#  Description:
#    Reads in a list of filenames, parses data into tagged sets, then
#    generates charts based on the user-selected tags
#
#    Function is based on the matplotlib and numpy libraries
#
#    Easily customizable for custom data processing/chart generation
#
#    Input data file format:   <tag>:<x_index>,<y_index>
#
#  Inputs:
#    filenames:    list of filename strings
#    tags:         list of data tags to be displayed
#    title:        chart title
#    maxY:         Max value on Y axis
#    output_file:  File to save chart to
#
#######################################################################
def generate_chart(filenames, tags, title, maxY, output_file):

	# set up the graph figure
	fig = plt.figure(figsize=(8.0, 8.0))
	
	# set up color selections for the number of entries in the legend
	n = len(filenames) * len(tags)
	color=iter(cm.rainbow(np.linspace(0,1,n)))

	# iterate through filelist
	for filename in filenames:

		# Parse data from filename to aid in labeling legend entries
		name = ''
		m = re.match('(.*)_data([0-9]+)x([0-9]+)\.txt', filename)
		if m:
			name = m.group(1).split('_')
			name = ' '.join(name) + ', ' + m.group(3) + ' Threads'
			name = name.title() 
		else:
			m = re.match('(.*)_data\.txt', filename)
			if m:
				name = ' '.join(m.group(1).split('_')).title()
		
		# open file 
		f = open('data/'+filename, 'r')
		text = f.readlines()

		# lists for gathering custom data groups
		display = []
		update = []
		steps = []
		kernel = []
		transfer = []

		# parse through each data entry and look for tags
		for line in text:
			line = line.split(':')
			tag = line[0]
			result = line[1]
			result = result.split(',')
			step = result[0]
			result = result[1]

			if int(step) > 0:
				if tag == 'display':
					display.append(float(result))
				elif tag == 'update':
					update.append(float(result))
					steps.append(int(step))
				elif tag == 'kernel':
					kernel.append(float(result))
				elif tag == 'transfer':
					transfer.append(float(result))

		# if tag was requested, and entries were found, add display to plot
		if 'display' in tags and display:
			c=next(color)
			plt.plot(steps, display, c=c, marker='x', label='Display Runtime, '+name)
		if 'update' in tags and update:
			c=next(color)
			plt.plot(steps, update, c=c, label='Simulation Runtime, '+name)
		if 'total' in tags and display and update:
			c=next(color)
			plt.plot(steps, map(add,update,display), c=c, label='Total Runtime, '+name)
		if 'kernel' in tags and kernel:
			c=next(color)
			plt.plot(steps, kernel, c=c, label='Kernel Runtime, '+name)
		if 'transfer' in tags and transfer:
			c=next(color)
			plt.plot(steps, transfer, c=c, marker='.', label='Transfer Runtime, '+name)

	# generate plot
	plt.ylabel('Runtime (ms)')
	plt.xlabel('Simulation Step')
	plt.title(title)
	plt.grid(True)
	plt.legend(loc='upper right')
	plt.ylim(0,maxY)
	plt.savefig(output_file)
# ----- END GENERATE_CHART ----- #

####################### MAIN PROGRAM #########################

# ----- INITIALIZATION ----- #
# lists for holding file groups
parallel_files = []
parallel_async_files = []
parallel_streaming_files = []

# counter used for simple filenames
chartCnt = 1; 

# Gather all simple parallel files
path = os.path.dirname(os.path.abspath(__file__)) + '/data'
for filename in next(os.walk(path))[2]:
	m = re.match('(life_parallel)_data([0-9]+)x([0-9]+)\.txt', filename)
	if m:
		parallel_files.append(m.group(0))

# Gather all async parallel files
path = os.path.dirname(os.path.abspath(__file__)) + '/data'
for filename in next(os.walk(path))[2]:
	m = re.match('(life_parallel_async)_data([0-9]+)x([0-9]+)\.txt', filename)
	if m:
		parallel_async_files.append(m.group(0))

# Gather all streaming parallel files
path = os.path.dirname(os.path.abspath(__file__)) + '/data'
for filename in next(os.walk(path))[2]:
	m = re.match('(life_parallel_streaming)_data([0-9]+)x([0-9]+)\.txt', filename)
	if m:
		parallel_streaming_files.append(m.group(0))

# ----- CHART GENERATION ----- #
generate_chart(['life_serial_data.txt'], ['total','display','update'], 
'Serial Implementation of Game of Life, 100 Steps', 4, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(parallel_files, ['kernel'], 
'Parallel Implementation of Game of Life, 100 Steps', .14, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_serial_data.txt', 'life_parallel_data48x256.txt'], ['update'], 
'Serial vs. Parallel, 100 Steps', 2, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_serial_data.txt', 'life_parallel_data48x256.txt'], ['kernel', 'transfer', 'update'], 
'Serial vs. Parallel, 100 Steps', 1, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(parallel_async_files, ['update'], 
'Asynchronous Parallel Implementation of Game of Life, 100 Steps', .22, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_serial_data.txt', 'life_parallel_data48x256.txt','life_parallel_async_data384x32.txt'], ['update'], 
'Serial vs. Async Parallel, 100 Steps', 1, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_parallel_data48x256.txt','life_parallel_async_data384x32.txt'], ['kernel','transfer'], 
'Parallel vs. Async Parallel, 100 Steps', 1, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(parallel_streaming_files, ['update'], 
'Asynchronous Streaming Implementation of Game of Life, 100 Steps', .22, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_parallel_async_data384x32.txt','life_parallel_streaming_data192x64.txt'], ['kernel','transfer'], 
'Async Parallel vs. Streaming Parallel, 100 Steps', .22, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_serial_data.txt', 
'life_parallel_data48x256.txt','life_parallel_async_data384x32.txt','life_parallel_streaming_data192x64.txt'], 
['update'], 'Comparison of All Methods, Sim Only, 100 Steps', 1, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_serial_data.txt', 'life_parallel_data48x256.txt','life_parallel_async_data384x32.txt','life_parallel_streaming_data192x64.txt'],
['total'], 'Comparison of All Methods, Total Runtime, 100 Steps', 5, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

generate_chart(['life_parallel_streaming_data192x64.txt'], ['kernel','transfer','display'], 
'Breakdown of Parallel Streaming, 100 Steps', 2, 'charts/chart'+str(chartCnt)+'.png')
chartCnt = chartCnt + 1

# END #

