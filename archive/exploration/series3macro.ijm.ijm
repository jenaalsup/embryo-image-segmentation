// ✅ Duplicate and isolate
run("Duplicate...", "title=Segmentation duplicate");
selectWindow("Segmentation");

// ✅ Focus on DAPI (channel 1 in Fiji = channel 0 in script)
Stack.setChannel(0);

// ✅ Clean projection prep
run("8-bit");
run("Z Project...", "projection=[Median]");
rename("Projected");

// ✅ Optional denoising and edge smoothing
run("Despeckle");      // removes high-frequency noise
run("Close");          // smooth jagged borders

// ✅ Fixed threshold (tweak between 60–90 based on visibility)
setThreshold(75, 255);
run("Convert to Mask");

// ✅ Fill internal gaps and apply softer watershed
run("Dilate");         // pad blobs slightly before split
run("Erode");          // undo the dilation shape-wise
run("Fill Holes");
run("Watershed");

// ✅ Particle analysis
run("Set Measurements...", "area centroid display redirect=None decimal=2");
roiManager("reset");
run("Analyze Particles...", "size=30-Infinity circularity=0.20-1.00 show=Overlay display clear include add");
roiManager("Deselect");
roiManager("Show All with labels");
