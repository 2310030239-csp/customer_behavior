CCTV Customer Behavior Analysis - Windows CMD Step-by-step

1) Prerequisites:
   - Python 3.9+ installed and python added to PATH.
   - Git (optional) if you clone repo.
   - This project uses OpenCV, numpy, pandas, matplotlib, scipy.

2) Extract project (if zipped). CD into the project folder, e.g.:
   cd C:\path\to\cctv_project

3) Create and activate virtual environment (Windows CMD):
   python -m venv venv
   venv\Scripts\activate

4) Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt

5) Run on the provided sample video:
   python main.py --input sample_footage.mp4 --output_dir output

   After completion, check the 'output' folder:
    - tracks.csv       (frame,id,x,y,w,h)
    - heatmap.png      (heatmap image)
    - trajectories.mp4 (video overlaying tracks)

6) To run on your own CCTV file:
   python main.py --input path\to\your_video.mp4 --output_dir output_yourvideo

7) Troubleshooting:
   - If OpenCV fails to open video, ensure the path is correct and codec is supported.
   - If CPU is bottleneck, reduce resolution or fps.

Notes on what the code does:
 - Uses BackgroundSubtractorMOG2 to detect moving foreground (people)
 - Filters small contours, computes bounding boxes
 - Simple centroid-based tracker associates detections frame-to-frame
 - Generates a heatmap by accumulating detection boxes and saves a heatmap image
 - Saves per-frame tracked centroids to CSV for downstream analytics (dwell time, zone entry)
