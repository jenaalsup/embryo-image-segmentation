/* 
  Script to segment slice of human embryo image and save attributes like aspect ratio, coordinates, and size of each nucleus
  Before running, open and select the correct, day, series, and z-sluce image
  Paste this code into a Fiji macro
*/

outputPath = "/Users/jenaalsup/Desktop/D3_series8-again.csv"; // change this to be the day / series of interest

// duplicate and isolate DAPI channel
run("Duplicate...", "duplicate channels=1 title=Segmentation duplicate"); // pass in DAPI (channels=1) or SOX2 (channels=5)selectWindow("Segmentation");

// projection prep
run("8-bit");

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
run("Set Measurements...", "area centroid shape display redirect=None decimal=2");
roiManager("reset");
run("Analyze Particles...", "size=30-300 circularity=0.00-1.00 show=Overlay display clear include add");
roiManager("Deselect");
roiManager("Show All with labels");

// save results to CSV (update the path and filename as needed)
saveAs("Results", outputPath);
