"""
tracker.py
Simple centroid tracker with basic matching by distance.
Stores bounding boxes, centroids, and trajectories.
"""
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # next object ID
        self.nextObjectID = 0
        self.objects = dict()            # objectID -> centroid (x,y)
        self.bboxes = dict()             # objectID -> (x,y,w,h)
        self.disappeared = dict()        # objectID -> frames disappeared
        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance
        self.trajectories = dict()       # objectID -> list of centroids

    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.trajectories[self.nextObjectID] = [tuple(centroid)]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
        del self.trajectories[objectID]

    def update(self, rects):
        # rects: list of (x,y,w,h)
        if len(rects) == 0:
            # mark disappeared
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.maxDisappeared:
                    self.deregister(oid)
            return self.objects

        inputCentroids = []
        for (x,y,w,h) in rects:
            cX = int(x + w/2.0)
            cY = int(y + h/2.0)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i,cent in enumerate(inputCentroids):
                self.register(cent, rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), np.array(inputCentroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (r,c) in zip(rows, cols):
                if r in usedRows or c in usedCols:
                    continue
                if D[r,c] > self.maxDistance:
                    continue
                objectID = objectIDs[r]
                self.objects[objectID] = inputCentroids[c]
                self.bboxes[objectID] = rects[c]
                self.disappeared[objectID] = 0
                self.trajectories[objectID].append(tuple(inputCentroids[c]))
                usedRows.add(r)
                usedCols.add(c)

            # check unused rows -> disappeared
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for r in unusedRows:
                objectID = objectIDs[r]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # register new input centroids not matched
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for c in unusedCols:
                self.register(inputCentroids[c], rects[c])

        return self.objects
