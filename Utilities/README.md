# Hints

Connect three videos side-by-side in the command line:

```
ffmpeg -i movie_1ch.mp4 -i movie_2ch.mp4 -i movie_3ch.mp4 -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -map 0:a? -map 0:a? -map 0:a? -c:v libx264 -preset medium -crf 23 -c:a aac -b:a 128k output_three_side_by_side.mp4
```
