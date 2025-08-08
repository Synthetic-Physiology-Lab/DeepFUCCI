# Cell tracking

You can in principle use any tracker.
We used [TrackMate](https://imagej.net/plugins/trackmate/) through Fiji.
The segmentation masks were added to the image as a separate channel.
Then, the label detector was used on the channel containing the segmentation masks.
The standard LAP tracker was used with settings that were manually adjusted to enable a good linking quality.
Importantly, we imposed a penalty on the intensity to leverage that FUCCI nuclei will not abruptly change their color.

**For a successful analysis, it is important to postprocess the tracks in TrackMate!**
The TrackMate [actions](https://imagej.net/plugins/trackmate/actions/) “Close gaps in tracks by introducing new spots” (https://imagej.net/plugins/trackmate/actions/close-gaps-action) and
“Auto naming spots” need to be used.
The autonaming option to append letters for each branch should be chosen
so that cell divisions can be detected from the spot name. 

Then, the tracking result should be exported as an XML file (few cells).
In case of many cells, rather the CSV format should be used and the spots
should be exported.

An example notebook to process the tracking data and visualize the result in Napari can be found in **TODO** (add when final figure format is ready).
Here, also the DTW distortion score is computed and cells are classified based on it.
More details can be found in the `fucciphase_processing` folder. 
