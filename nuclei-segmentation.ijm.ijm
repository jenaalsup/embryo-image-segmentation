/* Nuclei segmentation
   Tested on SHEF6_D2N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif, series 3, z-slice 40/79
*/

// duplicate and isolate DAPI channel
run("Duplicate...", "duplicate channels=1 title=Segmentation duplicate");
selectWindow("Segmentation");

// projection prep
run("8-bit");
run("Z Project...", "projection=[Median]");
rename("Projected");

// pre-threshold smoothing
run("Gaussian Blur...", "sigma=0.8");
run("Despeckle"); // remove high frequency noise
run("Close"); // smooth jagged borders

// apply fixed threshold
setThreshold(75, 255);
run("Convert to Mask");

// post-threshold smoothing
run("Dilate");
run("Fill Holes");
run("Median...", "radius=1"); // smooth edges while preserving shape
run("Watershed");

// particle analysis (labeling)
run("Set Measurements...", "area centroid display redirect=None decimal=2");
roiManager("reset");
run("Analyze Particles...", "size=30-300 circularity=0.00-1.00 show=Overlay display clear include add");
roiManager("Deselect");
roiManager("Show All with labels");
