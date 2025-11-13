### Inner cavity segmentation (with Napari/nninteractive)
---
1. Open Anaconda Prompt  
   (On Windows, click the Start button and search for "Anaconda Prompt")

2. Create a new environment  
   ```
   conda create -n nninteractive python=3.10 -y
   ```
   Note: If this generates an error related to retry failed, just rerun the command.
   
4. Activate the environment  
   ```
   conda activate nninteractive
   ```

5. Install Napari and the nninteractive plugin  
   ```
   pip install "napari[all]" napari-nninteractive
   ```

6. Launch Napari with the plugin  
   ```
   napari -w napari-nninteractive
   ```
   This opens the Napari GUI with the nninteractive plugin already open.

7. Drag the raw .tif file into the GUI (any number of lumens ok).

8. Press Initialize in the right-side panel. This will open an nninteractive label layer that looks like the screenshot below:  
   <img width="2171" height="1308" alt="Screenshot 2025-11-09 153413" src="https://github.com/user-attachments/assets/0349e7d0-f95e-4fe2-b7f9-465d590b0325" />

9. Under "Prompt type", select "Positive", and under "Interaction tools", select "Point".

10. Click inside the lumen cavities to place positive points (~1 per lumen or as needed).  
   When placing points, note that the cavity will be colored red; it will take ~4 seconds for this to show up.  
   Only if needed, switch the prompt type to "Negative" and place points outside of the cavity. This is to correct for overshoot (eg the cavity is bigger than intended).  
   Example slice of lumens after placing 1 positive point in the left lumen and 3 positive points in the right lumen:  
    <img width="2170" height="1300" alt="Screenshot 2025-11-09 154239" src="https://github.com/user-attachments/assets/6cb73586-c863-48ef-894d-ae22b20a1ba2" />

11. Go to File > Save Selected Layer(s) and choose the Labels layer.


### Outer cavity segmentation (with Napari/nninteractive)
---
- same steps as inner basically
- need roughly 4 clicks inside each lumen depending on size to get the whole thing filled
- save each lumen as a separate file. press "next object" in between each step so that they are all saved in different layers
- can also potentially do inner/outer at the same time by putting positive points in the actual lumen, negative in the cavity DONT DO THIS
- sometimes napari crashes when you want to do more than 2 lumens???
<img width="2154" height="1380" alt="image" src="https://github.com/user-attachments/assets/bee96b17-602b-4245-9c67-22b227256a71" />

### Ellipse fitting
---
1. Move all files outputted from the previous step to be in this file structure, where each file has its own folder and for n lumens, there are n + 1 files:
<img width="784" height="311" alt="Screenshot 2025-11-12 at 2 28 25 PM" src="https://github.com/user-attachments/assets/c4fac917-5910-42bd-aa61-dc54f354a446" />

### [deprecated] Outer cavity segmentation (with thresholding) - alternative to Napari
---
1. export IMAGE_PATH="..." (e.g. /Users/jenaalsup/Desktop/CKHRJQ~2.TIF)
2. python3 segment-outer.py (expect this to take ~2-3 minutes)

2. export SEGMENTATION_DIR="..." (e.g. /Users/jenaalsup/Desktop/segmentation-testing)
3. python3 image_analysis.py
