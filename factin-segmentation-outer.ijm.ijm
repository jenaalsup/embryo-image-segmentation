run("Duplicate...", "duplicate channels=4 title=MembraneSegmentation");
selectWindow("MembraneSegmentation");

// Project and clean
run("8-bit");
run("Z Project...", "projection=[Median]");
rename("MembraneProjected");

run("Gaussian Blur...", "sigma=1.2");
run("Despeckle");
run("Close");

// Threshold and binarize
setThreshold(20, 255);
run("Convert to Mask");
run("Fill Holes");
run("Open");

// Extract outer contour of the large object only
run("Analyze Particles...", "size=50-Infinity show=Masks clear add");
rename("CleanedMask");
run("Outline");
rename("OuterContour");

// Save as ROI
//roiManager("reset");
run("Create Selection");
roiManager("Add");
close();

// Overlay ROI onto original slice
selectWindow("SHEF6_D3N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif - Series003");
Stack.setSlice(28);
roiManager("Select", 0);
roiManager("Show All");
