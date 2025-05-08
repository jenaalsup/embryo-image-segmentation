// ✅ Duplicate channel 1 (DAPI only)
run("Duplicate...", "duplicate channels=1 title=Segmentation");
selectWindow("Segmentation");

// ✅ Convert to 8-bit and blur
run("8-bit");
run("Gaussian Blur 3D...", "x=1 y=1 z=1");

// ✅ Threshold and make binary
setThreshold(60, 255);
run("Convert to Mask");

// Clean up binary mask (slice-by-slice 2D morphology)
run("Dilate");
run("Erode");
run("Erode"); // run a second time to prevent watershed from being too aggressive
run("Fill Holes");
run("Watershed");

// ✅ Run 3D Objects Counter — get statistics + label map
run("3D Objects Counter", "threshold=1 min.=30 max.=1000000 show_object_map statistics");

// ✅ Prepare for 2D slice-by-slice labeling
selectWindow("Segmentation");
run("Remove Overlay");
setFont("SansSerif", 12, "bold");

// explicitly set threshold for Analyze Particles
run("Set Measurements...", "area mean centroid redirect=None decimal=2");
run("Clear Results"); // add before loop
setThreshold(128, 255);

stackSize = nSlices;
for (s = 1; s <= stackSize; s++) {
    setSlice(s);
	run("Analyze Particles...", "size=30-Infinity circularity=0.2-1.0 show=Nothing display overlay");
}

run("Show Overlay");

