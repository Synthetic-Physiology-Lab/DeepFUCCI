**Please refer to the following paper for more details:**
 Ghannoum, S.; Antos, K.; Leoncio Netto, W.; Gomes, C.; Köhn-Luque, A.; Farhan, H. CellMAPtracer: A User-Friendly Tracking Tool for Long-Term Migratory and Proliferating Cells Associated with FUCCI Systems. Cells 2021, 10, 469. https://doi.org/10.3390/cells10020469 

Download the data from: https://zenodo.org/records/4179252
The file containing raw data and segmentation masks is called:
`RGB-PIP-FUCCI.tif`

The first channel is the PCNA signal, the second channel
is the Geminin signal, the third channel is the PIP signal.
When the PIP signal is active, the cell is in G1, when
the Geminin signal is active, the cell is in S and when
both signals are active, the cell is in G2.

Before segmenting, check the StarDist model directory in the script
and then run:
 
```
python segment_nuclei.py
python view_in_napari.py
```

**Note: Here, the PCNA signal is used as the third channel and the
pretrained StarDist network is used to predict it.
Moreover, we scale the image to adjust it for the trained resolution.**

# Accuracy check

We manually curated one frame to check the accuracy.
TODO insert accuracy check
