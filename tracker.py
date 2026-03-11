from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        # Set max_cosine_distance to 0.10 as a starting point — it sits in the middle of the gap. If you see ID switches, lower it toward 0.08. If tracks fragment (same turkey gets multiple IDs), raise it toward 0.13.
        max_cosine_distance = 0.1
        nn_budget = 100

        encoder_model_filename = 'runs/turkey_reid_v5/best_model.pb' #'model_feature_extractor/mars-small128.pb' #'output/frozen_model.pb' #'model_feature_extractor/mars-small128.pb'          #mars-small128.pb  frozen_model.pb

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks([])
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks(detections)

    def update_tracks(self, detections):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id
            
            # ---- find matching detection and take its score ----
            score = 1.0  # default fallback
    
            if len(detections) > 0:
                best_iou = 0
                best_score = 1.0
    
                tx1, ty1, tx2, ty2 = bbox
    
                for det in detections:
                    dx1, dy1, dx2, dy2, dscore = det
    
                    # compute IoU
                    ix1 = max(tx1, dx1)
                    iy1 = max(ty1, dy1)
                    ix2 = min(tx2, dx2)
                    iy2 = min(ty2, dy2)
    
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_t = (tx2 - tx1) * (ty2 - ty1)
                    area_d = (dx2 - dx1) * (dy2 - dy1)
                    union = area_t + area_d - inter
    
                    iou = inter / union if union > 0 else 0
    
                    if iou > best_iou:
                        best_iou = iou
                        best_score = dscore
    
                score = best_score

            tracks.append(Track(id, bbox, score))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    score = None

    def __init__(self, id, bbox, score):
        self.track_id = id
        self.bbox = bbox
        self.score = score
