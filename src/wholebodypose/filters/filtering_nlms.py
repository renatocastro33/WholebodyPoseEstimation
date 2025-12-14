import cv2
import numpy as np
from padasip.filters import AdaptiveFilter,FilterNLMS

def initialize_adaptive_filter(num_keypoints,mu=0.9):
    filters = []
    for _ in range(num_keypoints):        
        filter_x = FilterNLMS(2, mu=mu,w="random")
        filter_y = FilterNLMS(2, mu=mu,w="random")
        filter_z = FilterNLMS(2, mu=mu,w="random")
        filters.append((filter_x, filter_y,filter_z))
    return filters

class KeypointMemory:
    def __init__(self, window=10):
        self.window = window
        self.last_frame = -np.inf
        self.last_pred_kp = None
        self.last_pred_score = 0

    def update(self, frame_idx, pred_kp, pred_score):
        self.last_frame = frame_idx
        self.last_pred_kp = pred_kp
        self.last_pred_score = pred_score

    def get(self, frame_idx):
        if frame_idx - self.last_frame <= self.window:
            return self.last_pred_kp, self.last_pred_score
        else:
            return None, None


def track_keypoints(t, keypoints, scores, adaptive_filters, memories,
                    low=0.25, high=0.5, interp_alpha=0.5):
    keypoints_pred = np.zeros_like(keypoints)
    scores_pred    = np.zeros_like(scores)

    for i in range(len(keypoints)):
        x, y = keypoints[i]
        score = scores[i]
        filter_x, filter_y, filter_s = adaptive_filters[i]
        memory = memories[i]

        score_vec = score
        if score>high:
            score_vec = 1
        else:
            score_vec = score/high
        input_vec = np.array([t, score_vec])#int(score > high)])

        # Predicción del filtro
        pred_x = filter_x.predict(x=input_vec)
        pred_y = filter_y.predict(x=input_vec)
        pred_s = filter_s.predict(x=input_vec)

        if score >= high:
            # Confianza alta → usar valor observado
            filtered_x, filtered_y, filtered_s = x, y, score

        elif score < low:
            # Confianza baja → intentar usar memoria si existe
            remembered_kp, remembered_score = memory.get(t)
            if remembered_kp is not None:
                filtered_x, filtered_y = remembered_kp
                filtered_s = max(0.0, remembered_score - 0.05)
            else:
                # Sin memoria válida → usar predicción
                filtered_x, filtered_y, filtered_s = 0,0,0 #pred_x, pred_y, pred_s

        else:
            # Score medio → interpolar entre predicción y observación
            filtered_x = interp_alpha * x + (1 - interp_alpha) * pred_x
            filtered_y = interp_alpha * y + (1 - interp_alpha) * pred_y
            filtered_s = interp_alpha * score + (1 - interp_alpha) * pred_s
            print("x:",x,"pred_x:",pred_x,"filtered_x:",filtered_x)
        # Adaptar el filtro con observación (x, y, score), no el filtrado
        filter_x.adapt(x=input_vec, d=x)
        filter_y.adapt(x=input_vec, d=y)
        filter_s.adapt(x=input_vec, d=score)

        # ✅ Solo actualizar la memoria si el score no es muy bajo
        if filtered_s >= low:
            memory.update(t, [filtered_x, filtered_y], filtered_s)

        # Guardar resultados
        keypoints_pred[i] = [filtered_x, filtered_y]
        scores_pred[i] = filtered_s

    return keypoints_pred, scores_pred


class FilteringNLMS:
    def __init__(self,num_keypoints=135):
        self.num_keypoints = num_keypoints

    def apply(self, keypoints, scores):
        adaptive_filters = initialize_adaptive_filter(self.num_keypoints,mu=1.05)
        memories = [KeypointMemory(window=5) for _ in range(self.num_keypoints)]
        # FILTERING
        list_keypoints_new = []
        list_scores_new = []
        for cnt in range(keypoints.shape[0]):
            keyp_pred,scores_pred  = track_keypoints(cnt,keypoints[cnt,:,:],scores[cnt,:], adaptive_filters,memories)            
            list_keypoints_new.append(keyp_pred)
            list_scores_new.append(scores_pred)
        
        keypoints_new = np.array(list_keypoints_new)
        scores_new = np.array(list_scores_new)
        return keypoints_new, scores_new