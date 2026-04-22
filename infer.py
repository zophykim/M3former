import argparse
import os
import torch
import json
import numpy as np
import logging
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_AIS
from model import M3former
from types import SimpleNamespace
from utils.tools import vali_test
import torch.nn as nn
# =======================
# Logging with Colors
# =======================

class Color:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


logging.basicConfig(
    level=logging.INFO,
    format=f"{Color.BLUE}[%(levelname)s]{Color.END} %(message)s"
)

def banner(text):
    logging.info(f"{Color.BOLD}{Color.CYAN}\n========== {text} =========={Color.END}")

def ok(text):
    logging.info(f"{Color.GREEN}[OK]{Color.END} {text}")

def warn(text):
    logging.warning(f"{Color.YELLOW}{text}{Color.END}")

def error(text):
    logging.error(f"{Color.RED}{text}{Color.END}")


# =======================
# Core Functions
# =======================

def load_settings(checkpoint_path):
    path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        return SimpleNamespace(**data)
    raise FileNotFoundError(f"No settings/config file found under {checkpoint_path}")


def load_checkpoint_for_inference(model, checkpoint_path, device="cuda"):
    # allow specifying either a directory containing checkpoint.pth or a full file path
    if os.path.isdir(checkpoint_path):
        # search for files in directory
        # prefer explicit 'best_checkpoint.pth' or 'checkpoint.pth'
        for name in ["best_checkpoint.pth", "checkpoint.pth"]:
            candidate = os.path.join(checkpoint_path, name)
            if os.path.exists(candidate):
                checkpoint_file = candidate
                break
        else:
            # fallback to the first .pth file in directory
            files = [f for f in os.listdir(checkpoint_path) if f.endswith('.pth')]
            if not files:
                raise FileNotFoundError(f"No .pth file found in {checkpoint_path}")
            checkpoint_file = os.path.join(checkpoint_path, files[0])
    else:
        checkpoint_file = checkpoint_path
    ckpt = torch.load(checkpoint_file, map_location=device)

    # compatible with various formats
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif 'model' in ckpt:
            state = ckpt['model']
        else:
            # assume full state_dict
            state = ckpt
        model.load_state_dict(state)
        data_scaler = ckpt.get("data_scaler", None)
    else:
        # ckpt might just itself be a state_dict
        model.load_state_dict(ckpt)
        data_scaler = None

    model.to(device)
    model.eval()
    return model, data_scaler

def collate_fn(batch):
    seq_x = torch.stack([b[0] for b in batch])
    seq_y = torch.stack([b[1] for b in batch])
    seq_x_mark = torch.stack([b[2] for b in batch])
    seq_y_mark = torch.stack([b[3] for b in batch])
    static = [b[4] for b in batch]
    stats = [b[5] for b in batch]
    return seq_x, seq_y, seq_x_mark, seq_y_mark, static, stats


def batch_inverse_transform(arr, scaler, dims=None):
    """
    arr: (..., 2)
    scaler: sklearn MinMaxScaler fitted on 7 dims
    dims: 预测维度在原特征中的位置
    """
    shape = arr.shape
    flat = arr.reshape(-1, arr.shape[-1])  # (N, 2)

    if dims:
        full = np.zeros((flat.shape[0], scaler.n_features_in_))
        full[:, dims] = flat

        inv_full = scaler.inverse_transform(full)

        inv = inv_full[:, dims]
    else:
        inv = scaler.inverse_transform(flat)
    return inv.reshape(shape)

def metrics(pred, true):
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true)**2))
    return mae, rmse


# =======================
# Inference Procedure
# =======================

def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banner("Starting Inference")
    logging.info(f"Using device: {device}")
    logging.info(f"Loading settings from: {args.checkpoint}")

    # -------- Load settings.json --------
    margs = load_settings(args.checkpoint)

    # -------- Create model --------
    banner("Loading Model")
    model = M3former.Model(margs).to(device)


    # -------- load checkpoint --------
    logging.info(f"Loading checkpoint from: {args.checkpoint}")
    model, data_scaler = load_checkpoint_for_inference(model, args.checkpoint)
    ok("Checkpoint loaded successfully")

    # -------- dataset size & features --------
    size = (args.seq_len, args.label_len, args.pred_len)
    features = args.features_list.split(',') if args.features_list else \
               ['Longitude','Latitude','SOG','COG','Heading']
    static_features = args.static_features_list.split(',') if args.static_features_list else \
               ['MMSI','Ship type','Width','Length','Draught','Destination','ETA']


    # -------- Prepare dataset --------
    banner("Preparing Dataset")
    logging.info(f"test_csv  = {args.test_csv}")

    logging.info(f"seq_len={args.seq_len}, label_len={args.label_len}, pred_len={args.pred_len}")
    logging.info(f"features = {features}")
    logging.info(f"static   = {static_features}")

    test_data = Dataset_AIS(args.test_csv, size=size, features=features,
                            static_features=static_features, scaler=data_scaler)
    test_loader = DataLoader(test_data,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=False,num_workers=args.num_workers,drop_last=False)


    # ===== 验证 =====
    criterion = nn.MSELoss()
    mae_metric = nn.L1Loss()
    val_loss, _, step_errors = vali_test(args, model, device, test_data, test_loader, criterion, mae_metric, data_scaler)
    print(f"Test Loss: {val_loss:.6f}")

    test_name = os.path.splitext(os.path.basename(args.test_csv))[0]
    step_errors_path = os.path.join(args.checkpoint, f'step_errors_{test_name}.json')
    with open(step_errors_path, 'w') as f:
        json.dump(step_errors, f)
    ok(f"Step errors saved to {step_errors_path}")
    return val_loss, _, step_errors

# =======================
# CLI
# =======================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--output_file", type=str, default="temp.csv")

    parser.add_argument("--seq_len", type=int, default=36)
    parser.add_argument("--label_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=24)

    parser.add_argument("--features_list", type=str, default="")
    parser.add_argument("--static_features_list", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    inference(args)


#python infer.py --checkpoint 'checkpoint/2025-12-05 06:38:08long_term_forecast_AIS_24_12_12_AISLLM_AISDMA_ftM_sl24_ll12_pl12_dm32_nh8_el2_dl1_df32_fc1_ebtimeF_test_0-AISLLM-DMA' --test_csv 'dataset/AISDMA/test.csv' --train_csv 'dataset/AISDMA/train.csv'

# mixst textmode1 turnloss false
# traisformerQ_brisk-sweep-13
# python run_infer.py --checkpoint 'checkpoint/traisformerQ_brisk-sweep-13' --test_csv 'dataset/AISDMA/test.csv'