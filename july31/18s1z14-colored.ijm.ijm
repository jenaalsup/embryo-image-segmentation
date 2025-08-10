// -----------------------------
// Nuclear segmentation + color by aspect ratio
// Green = round (low AR) -> Cyan = medium AR -> Magenta = elongated (high AR)
// Saves measurements CSV and creates an RGB image "AR_colored"
// -----------------------------
outputPath = "/Users/jenaalsup/Desktop/18s1z14-segmentation.csv"; // Updated for new image
// Duplicate and isolate DAPI channel
run("Duplicate...", "duplicate channels=1 title=Segmentation duplicate"); // Updated to match your new method
selectWindow("Segmentation");
// Preprocessing
run("8-bit");
run("Gaussian Blur...", "sigma=0.8");
run("Despeckle"); // remove high frequency noise
run("Close"); // smooth jagged borders
// Threshold — fixed threshold (adjust if needed)
setThreshold(75, 255);
run("Convert to Mask");
// Morph cleanup — using your new method
run("Dilate");
run("Median...", "radius=1"); // smooth edges while preserving shape
run("Watershed");
// Prepare measurements (include 'fit' so Major/Minor are measured)
run("Set Measurements...", "area centroid shape fit redirect=None decimal=2");
// Analyze particles and add ROIs to manager
roiManager("reset");
run("Analyze Particles...", "size=30-300 circularity=0.00-1.00 show=Nothing display clear include add"); // Updated size range
roiManager("Deselect");
// Number of ROIs
n = roiManager("count");
if (n == 0) {
    print("No ROIs found. Exiting.");
    exit();
}
// Prepare arrays and min/max
aspectRatios = newArray(n);
minAR = 1e9;
maxAR = -1e9;
// Clear results then measure each ROI to build measurements table and compute AR
run("Clear Results");
for (i = 0; i < n; i++) {
    roiManager("select", i);
    run("Measure");
    idx = nResults - 1;
    major = getResult("Major", idx);
    minor = getResult("Minor", idx);
    if (minor > 0) {
        ar = major / minor;
    } else {
        ar = 1.0; // safe fallback
    }
    aspectRatios[i] = ar;
    if (ar < minAR) minAR = ar;
    if (ar > maxAR) maxAR = ar;
}
// Save measurements to CSV
saveAs("Results", outputPath);
// Create an RGB image to paint colored nuclei
w = getWidth();
h = getHeight();
newImage("AR_colored", "RGB black", w, h, 1);
selectWindow("AR_colored");
// Avoid division by zero if all AR equal
sameRange = (maxAR <= minAR + 1e-12);
// Paint each ROI on the RGB image with a color mapped from AR
for (i = 0; i < n; i++) {
    if (sameRange) {
        norm = 0.5;
    } else {
        norm = (aspectRatios[i] - minAR) / (maxAR - minAR);
        if (norm < 0) norm = 0;
        if (norm > 1) norm = 1;
    }
    
    // Three-color gradient: Green -> Cyan -> Magenta
    // Green (0,255,0) -> Cyan (0,255,255) -> Magenta (255,0,255)
    if (norm < 0.5) {
        // First half: Green to Cyan
        t = norm * 2; // 0 to 1
        r = 0;
        g = 255;
        b = round(t * 255); // 0 to 255
    } else {
        // Second half: Cyan to Magenta
        t = (norm - 0.5) * 2; // 0 to 1
        r = round(t * 255); // 0 to 255
        g = round((1 - t) * 255); // 255 to 0
        b = 255;
    }
    
    roiManager("select", i);
    setForegroundColor(r, g, b);
    roiManager("Fill");
    print("ROI " + i + ": AR=" + aspectRatios[i] + " norm=" + norm + " RGB(" + r + "," + g + "," + b + ")");
}
// Set ROI display properties to white outlines and labels
roiManager("Set Color", "white");
roiManager("Set Line Width", 1);
// Show segmentation with white labels on top
roiManager("Show All with labels");
selectWindow("AR_colored");
resetMinAndMax();
