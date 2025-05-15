selectWindow("SHEF6_D3N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif - Series003");
Stack.setSlice(28);

// duplicate factin channel 
run("Duplicate...", "duplicate channels=4 title=ToBeCropped");

// crop using bounded rectangle on offset center
selectWindow("ToBeCropped");
getBoundingRect(x, y, w, h);
cx = x + w / 2 - 25;
cy = y + h / 2 - 65;
crop_size = 400;
x0 = cx - crop_size / 2;
y0 = cy - crop_size / 2;
makeRectangle(x0, y0, crop_size, crop_size);
run("Crop");

// segmentation transformations
setAutoThreshold("Huang dark");
run("Convert to Mask");
run("Median...", "radius=1");
run("Convert to Mask");
run("Erode");
run("Dilate");
run("Convert to Mask");

// keep only the largest object
run("Analyze Particles...", "size=50-Infinity show=Nothing clear include add limit=1");
run("Options...", "iterations=4 count=1 black edm=8-bit do=Erode");
run("Options...", "iterations=5 count=1 black edm=8-bit do=Open");   // removes small junk
run("Options...", "iterations=5 count=1 black edm=8-bit do=Close");  // smooths edges
run("Gaussian Blur...", "sigma=1");
setAutoThreshold("Huang dark");
run("Convert to Mask");

// create skeleton
run("Skeletonize");
run("Analyze Skeleton (2D/3D)", "prune show=tagged save");
waitForUser("Press OK once 'Tagged skeleton' is open.");
selectWindow("Tagged skeleton");
setThreshold(1, 255);  // include all label values
run("Convert to Mask");
run("Create Selection");  // selects all visible non-black pixels
roiManager("reset");  
roiManager("Add");
close();

// overlay inner wall translated back to the starting position
selectWindow("SHEF6_D3N20-3dE_G3_DAPI_Factin-488_Bry-568_Sox2-647.lif - Series003");
Stack.setSlice(28);
roiManager("Select", 0);
roiManager("Translate", x0, y0);
roiManager("Show All with labels");
