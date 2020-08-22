import os
import ffmpy

inputdir = '/home/mhtl/Projects/caption-generation'


for filename in os.listdir(inputdir): 
	actual_filename = filename[:-4] 
	if(filename.endswith(".mp4")): 
		os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}.wav'.format(filename, actual_filename)) 
	else: 
		continue

