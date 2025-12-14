import torch
from torch.nn.utils.rnn import pad_sequence

PAD_IDX = 0
MAX_SEQ_TRAINING = 50  

def pad_sequence_gcn(keypoints):
    fixed_len = 50
    batch_size = len(keypoints)
    device = keypoints[0].device
    dtype = keypoints[0].dtype
    padded = torch.full((batch_size, fixed_len, 135, 2), PAD_IDX, device=device, dtype=dtype)

    for i, seq in enumerate(keypoints):
        L = seq.size(0)
        if L >= fixed_len:
            start = torch.randint(0, L - fixed_len + 1, (1,), device=device).item()
            padded[i] = seq[start:start + fixed_len]
        else:
            padded[i, :L] = seq

    return padded

def apply_padding(keypoints):
    keypoints = pad_sequence_gcn(keypoints)  # [B, T, 135, 2]
    keypoints = keypoints.flatten(start_dim=2).float()  # [B, T, 270]
    if keypoints.shape[1] > MAX_SEQ_TRAINING:
        keypoints = keypoints[:, :MAX_SEQ_TRAINING, :]
    padding_mask = (keypoints == PAD_IDX).all(dim=2)  # [B, T]
    return keypoints, padding_mask

def collate_fn_gcnbert(batch):
    keypoints, names, labels, _ = zip(*batch)  # each keypoint: [T, 135, 2]
    keypoints, padding_mask = apply_padding(list(keypoints))  # [B, T, 270], [B, T]
    labels = torch.tensor(labels, dtype=torch.long)
    return keypoints, labels, padding_mask, names