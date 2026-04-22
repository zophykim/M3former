import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStoppingV2:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path, optimizer=None, scheduler=None, scaler=None, data_scaler=None, epoch=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, data_scaler, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            msg = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.accelerator:
                self.accelerator.print(msg)
            else:
                print(msg)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path, optimizer, scheduler, scaler, data_scaler, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, optimizer, scheduler, scaler, data_scaler, epoch):
        if self.verbose:
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...'
            if self.accelerator:
                self.accelerator.print(msg)
            else:
                print(msg)

        if self.accelerator:
            model = self.accelerator.unwrap_model(model)

        save_path = os.path.join(path, "checkpoint.pth")

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None,
            "data_scaler": data_scaler,         # ⭐ 新增
            "epoch": epoch,
            "val_loss_min": val_loss,
        }

        torch.save(checkpoint, save_path)
        self.val_loss_min = val_loss



class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def loss_nail(true,pred):

    # 分别取 lat / lon
    pred_lat = pred[:, :, 0]
    pred_lon = pred[:, :, 1]
    true_lat = true[:, :, 0]
    true_lon = true[:, :, 1]

    # 经纬度差
    dlat = pred_lat - true_lat          # [N, T]
    dlon = pred_lon - true_lon          # [N, T]

    # 纬度转弧度
    lat_rad = np.radians(true_lat)      # [N, T]

    # 每度对应的海里
    NM_PER_DEG_LAT = 60.0
    NM_PER_DEG_LON = 60.0 * np.cos(lat_rad)

    # 换算海里
    lat_dist_nm = dlat * NM_PER_DEG_LAT
    lon_dist_nm = dlon * NM_PER_DEG_LON

    # 每个点的误差
    dist_nm = np.sqrt(lat_dist_nm**2 + lon_dist_nm**2)   # [N, T]

    # ------ 单步 MAE / RMSE ------
    for t in [5, 11]:    # 第6步=索引5; 第12步=索引11
        mae_t = dist_nm[:, t].mean()
        rmse_t = np.sqrt((dist_nm[:, t]**2).mean())
        print(f"Step {t+1}: MAE = {mae_t:.4f} NM, RMSE = {rmse_t:.4f} NM")


def batch_inverse_transform(arr, scaler, dims=None):
    """
    arr: (..., k)
    scaler: FixedMinMaxScaler
    dims: arr 中对应原特征的索引
    """
    shape = arr.shape
    flat = arr.reshape(-1, arr.shape[-1])

    if dims is not None:
        full = np.zeros((flat.shape[0], scaler.n_features_in_))
        full[:, dims] = flat

        inv_full = scaler.inverse_transform(full)
        inv = inv_full[:, dims]
    else:
        inv = scaler.inverse_transform(flat)

    return inv.reshape(shape)


# def batch_inverse_transform(arr, scaler, dims=None):
#     """
#     arr: (..., 2)
#     scaler: sklearn MinMaxScaler fitted on 7 dims
#     dims: 预测维度在原特征中的位置
#     """
#     shape = arr.shape
#     flat = arr.reshape(-1, arr.shape[-1])  # (N, 2)

#     if dims:
#         full = np.zeros((flat.shape[0], scaler.n_features_in_))
#         full[:, dims] = flat

#         inv_full = scaler.inverse_transform(full)

#         inv = inv_full[:, dims]
#     else:
#         inv = scaler.inverse_transform(flat)
#     return inv.reshape(shape)

def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric,data_scaler):
    total_loss = []
    total_mae_loss = []
    all_pred_list = []
    all_true_list = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, static, stats) in tqdm(enumerate(vali_loader)):
            
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, stats)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, stats)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, stats)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, stats)

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            outputs, batch_y = accelerator.gather_for_metrics((outputs[:,:,:2], batch_y[:,:,:2]))



            # pred = outputs
            # true = batch_y[:,:,:2]
            true = batch_inverse_transform(batch_y.cpu().numpy(), data_scaler, dims=(0, 1))
            pred = batch_inverse_transform(outputs.cpu().numpy(), data_scaler, dims=(0, 1))

            all_pred_list.append(pred)
            all_true_list.append(true)
        
            loss = criterion(outputs, batch_y)

            mae_loss = mae_metric(outputs, batch_y)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    loss_nail(np.concatenate(all_true_list, axis=0),np.concatenate(all_pred_list, axis=0))
            

    model.train()
    return total_loss, total_mae_loss


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                stats,
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content


from model import M3former



def vali_test(args, model, device, vali_data, vali_loader, criterion, mae_metric,data_scaler):
    total_loss = []
    total_mae_loss = []
    all_pred_list = []
    all_true_list = []
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, static, stats) in tqdm(enumerate(vali_loader)):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)


            outputs = M3former.sample(model,
                    batch_x[:,:,:4],
                    stats,
                    args.pred_len,
                    temperature=1.0,
                    sample=True,
                    sample_mode='pos_vicinity',
                    r_vicinity=40,
                    top_k=10)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :]

            true = batch_inverse_transform(batch_y[:,:,:2].cpu().numpy(), data_scaler, dims=(0, 1))
            pred = batch_inverse_transform(outputs[:,:,:2].cpu().numpy(), data_scaler, dims=(0, 1))

            all_pred_list.append(pred)
            all_true_list.append(true)
        
            loss = criterion(outputs[:,:,:2], batch_y[:,:,:2])

            mae_loss = mae_metric(outputs[:,:,:2], batch_y[:,:,:2])

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    loss_nail(np.concatenate(all_true_list, axis=0),np.concatenate(all_pred_list, axis=0))
            

    model.train()
    return total_loss, total_mae_loss

    