ffmpeg -framerate 30 -i newframes/%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p <output-file.mp4>
