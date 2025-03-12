**Please refer to the following paper for more details:**
Cotton, M.J., Ariel, P., Chen, K. et al. An in vitro platform for quantifying cell cycle phase lengths in primary human intestinal epithelial cells. Sci Rep 14, 15195 (2024). https://doi.org/10.1038/s41598-024-66042-9

Download the data from: https://doi.org/10.5281/zenodo.11506667
The file containing raw data and segmentation masks is called:
`MJC004 Pos1 Control 4C.tif`

The video contains empty frames, thus it is first processed by running:

```
python split_published_data.py 
```

Then, check the StarDist model directory in the script
and run:
 
```
python segment_nuclei.py
python view_in_napari.py
```
