# Comparison with ConfluentFUCCI

We compare against the custom-trained Cellpose model
provided by the ConfluentFUCCI authors 
(paper can be found [here](https://doi.org/10.1371/journal.pone.0305491)).
The authors trained Cellpose on each FUCCI channel separately.
Thus, we segment each channel separately and compare against our network.

The authors have reported that ConfluentFUCCI massively outperforms
classic methods such as [FUCCItrack](https://doi.org/10.1371/journal.pone.0268297).
For this reason, we did not compare our approach against other appraoches.

To use the ConfluentFUCCI models, make sure that Cellpose is installed.
Download their models from [GitHub](https://github.com/leogolds/ConfluentFUCCI/tree/main/models/cellpose)
and deposit them in the `~/.cellpose/models` folder.

Edit (or create) the file `~/.cellpose/models/gui_models.txt` so that it contains the following lines:

```
nuclei_green_v2
nuclei_red_v2
```

Now, you can use their model.

Run `confluent_fucci_on_validation_data.py` to check the performance
on our validation data.


**Note:**

The authors used Cellpose v2, so you must install cellpose < 4.
In addition, the reference pixel size was not specified.
The reference diameter was 18 pixels.
We found that dividing the reference diameter with our pixel size
yielded a good segmentation, so probably the authors had
a reference pixel size of 1 micron.

## Validation

We used all of our pre-trained networks because the others provided
also a third channel, the brightfield channel.
In the paper, we only report the 2-channel result.
