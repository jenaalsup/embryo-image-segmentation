// Duplicate and extract membrane channel
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

// Extract largest object
run("Analyze Particles...", "size=500-Infinity show=Nothing clear add");
roiManager("Select", 0);
run("Create Mask");
rename("LargestObject");

// Get outer contour
run("Outline");
rename("OuterContour");
run("Create Selection");
roiManager("Add");

// Export outline to CSV
roiManager("Select", roiManager("Count") - 1);
run("Save XY Coordinates...", "path=/Users/jenaalsup/Desktop/ellipse-contour.csv");  // ⬅️ change this
// Duplicate and extract membrane channel
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

// Extract largest object
run("Analyze Particles...", "size=500-Infinity show=Nothing clear add");
roiManager("Select", 0);
run("Create Mask");
rename("LargestObject");

// Get outer contour
run("Outline");
rename("OuterContour");
run("Create Selection");
roiManager("Add");

// Export outline to CSV
roiManager("Select", roiManager("Count") - 1);
run("Save XY Coordinates...", "path=/Users/jenaalsup/Desktop/ellipse-contour.csv");  // ⬅️ change this
