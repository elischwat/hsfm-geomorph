conda activate hsfm
unset DISPLAY # dem_align.py will try to bring up the final plot for viewing. this is a way to disable that
nohup python routine.py & # to send the process to the background
tail -f nohup.out # to watch the log
htop # to watch the processing