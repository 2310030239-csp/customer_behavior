"""
main.py
Customer behavior analysis on CCTV footage (background subtraction + centroid tracking)

Produces:
 - tracks.csv         (frame,id,x,y,w,h)
 - counts per frame printed to console
 - heatmap.png        (heatmap of detections)
 - trajectories.mp4   (video overlaying tracked IDs and trajectories)

Usage:
    python main.py --input sample_footage.mp4
"""

import cv2
import numpy as np
import argparse
import os
from tracker import CentroidTracker
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='path to input video')
    p.add_argument('--output_dir', default='output', help='directory to save outputs')
    p.add_argument('--min_area', type=int, default=300, help='min contour area to consider a person')
    return p.parse_args()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    tracker = CentroidTracker(maxDisappeared=20, maxDistance=60)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_traj = cv2.VideoWriter(os.path.join(args.output_dir,'trajectories.mp4'), fourcc, fps, (w,h))
    heatmap_accum = np.zeros((h,w), dtype=np.float32)

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg = backSub.apply(frame)
        # morphological clean
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for c in contours:
            x,y,wc,hc = cv2.boundingRect(c)
            if wc*hc < args.min_area:
                continue
            rects.append((x,y,wc,hc))
            # accumulate for heatmap
            heatmap_accum[y:y+hc, x:x+wc] += 1

        objects = tracker.update(rects)

        # draw
        vis = frame.copy()
        for (objectID, centroid) in objects.items():
            # get bbox stored in tracker
            bbox = tracker.bboxes.get(objectID, None)
            if bbox is not None:
                x,y,wc,hc = bbox
                cv2.rectangle(vis, (x,y), (x+wc, y+hc), (0,255,0), 2)
            cv2.putText(vis, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            # draw trajectory
            pts = tracker.trajectories.get(objectID, [])
            for i in range(1, len(pts)):
                cv2.line(vis, tuple(pts[i-1]), tuple(pts[i]), (0,200,200), 2)

            rows.append({'frame':frame_idx,'id':objectID,'x':centroid[0],'y':centroid[1],
                         'w':bbox[2] if bbox else 0,'h':bbox[3] if bbox else 0})

        # overlays: count
        cv2.putText(vis, f"Count: {len(objects)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,0),2)

        out_traj.write(vis)
        frame_idx += 1

    cap.release()
    out_traj.release()

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir,'tracks.csv'), index=False)

    # heatmap
    hm = heatmap_accum
    # normalize and save heatmap using matplotlib
    plt.figure(figsize=(8,5))
    plt.imshow(hm, cmap='hot', interpolation='nearest', origin='upper')
    plt.title('Detection Heatmap')
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir,'heatmap.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    print('Finished. Outputs written to', args.output_dir)

if __name__=='__main__':
    main()
