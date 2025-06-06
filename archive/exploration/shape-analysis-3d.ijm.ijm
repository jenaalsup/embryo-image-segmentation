// ✅ Duplicate channel 1 (DAPI only)
run("Duplicate...", "duplicate channels=1 title=Segmentation");
selectWindow("Segmentation");

// ✅ Convert to 8-bit and blur
run("8-bit");
run("Gaussian Blur 3D...", "x=1 y=1 z=1");

// ✅ Threshold and make binary
setThreshold(60, 255);
run("Convert to Mask");

// ✅ Clean up
run("Dilate");
run("Erode");
run("Erode");
run("Fill Holes");
run("Watershed");

// ✅ 3D Objects Counter (for future use, optional)
run("3D Objects Counter", "threshold=1 min.=30 max.=1000000 show_object_map statistics");

// ✅ Clear overlay and set font
selectWindow("Segmentation");
run("Remove Overlay");
setFont("SansSerif", 12, "bold");

// ✅ Enable extended shape features
run("Set Measurements...", "area mean centroid fit shape redirect=None decimal=2");
run("Clear Results");
setThreshold(128, 255);

// ✅ Analyze each slice and calculate aspect ratio etc
stackSize = nSlices;
for (s = 1; s <= stackSize; s++) {
    setSlice(s);
    run("Analyze Particles...", "size=30-Infinity circularity=0.2-1.0 show=Nothing display overlay");
}

run("Show Overlay");

// ✅ Save results table (update path!)
saveAs("Results", "/Users/jenaalsup/Desktop/nuclei_shape_results.csv");
