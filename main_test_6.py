from utils import *
from evaluator import Evaluator
from test_model_1 import ProductManifoldGNN
from sklearn.utils import shuffle
import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import math
from tqdm import tqdm
import torch_geometric
import argparse
import numpy as np
import datetime
import pickle
from qm9_preprocess import get_dataset
import warnings
from sklearn.model_selection import train_test_split
import shutil
import time
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

cls_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=None) 
cls_criterion_1 = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.L1Loss()
mse_reg_criterion = torch.nn.MSELoss()
cosine_criterion = torch.nn.CosineEmbeddingLoss()

def compute_atom_refs(dataset, target_idx, max_z=100):
    print(f"Computing AtomRefs for target index {target_idx}...")
    
    all_z = dataset.data.x[:, 0].long()
    all_batch = dataset.data.batch
    if all_batch is None:
        print("Warning: Direct batch access failed, iterating dataset (slower)...")
        atom_counts = []
        targets = []
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            z = data.x[:, 0].long()
            counts = torch.bincount(z, minlength=max_z)
            atom_counts.append(counts)
            targets.append(data.y[0, target_idx].item())
        
        A = torch.stack(atom_counts).float()
        y = torch.tensor(targets).float()
    else:
        y = dataset.data.y[:, target_idx]
        num_mols = y.size(0)
        A = torch.zeros(num_mols, max_z, device=y.device)
        ones = torch.ones_like(all_z, dtype=torch.float)
        pass 

    try:
        sol = torch.linalg.lstsq(A, y).solution
    except:
        sol = torch.lstsq(y.unsqueeze(1), A).solution.squeeze()
        
    print("AtomRefs computed successfully.")
    return sol

def train(model, device, loader, optimizer, task_type, dataset_name, ema_model=None):
    model.train()
    loss_curve = list()
    
    align_criterion = torch.nn.MSELoss()
    lambda_align = 0.5 

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred, s0, hire_x_proj = model(batch)
            optimizer.zero_grad()

            if batch.y.shape != pred.shape:
                batch.y = batch.y.view(pred.shape)

            is_labeled = batch.y == batch.y
            
            task_loss = 0
            if "classification" in task_type:
                task_loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif "mse_regression" in task_type:
                target = batch.y.to(torch.float32)
                if hasattr(model, 'mean') and hasattr(model, 'std'):
                    target = (target - model.mean) / model.std
                
                if dataset_name == 'qm9':
                    task_loss = reg_criterion(pred.to(torch.float32)[is_labeled], 
                              batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))
                else:
                    task_loss = mse_reg_criterion(pred.to(torch.float32)[is_labeled],
                                                  target[is_labeled].view(pred[is_labeled].size()))
            
            consistency_loss = align_criterion(s0, hire_x_proj)
            loss = task_loss + lambda_align * consistency_loss

            if torch.isnan(loss):
                print(f"❌ Error: Loss is NaN at step {step}!")
                return float('nan')
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            if ema_model is not None:
                ema_model.update_parameters(model)

            loss_curve.append(loss.detach().cpu().item())

    return sum(loss_curve)/len(loss_curve)


@torch.no_grad()
def eval(model, device, loader, evaluator, task_type, dataset_name):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    loss_curve = list()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            pred, _, _ = model(batch)

            if batch.y.shape != pred.shape:
                batch.y = batch.y.view(pred.shape)

            is_labeled = batch.y == batch.y
            
            if "classification" in task_type:
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif task_type == 'mse_regression':
                if dataset_name == 'qm9':
                    loss = reg_criterion(pred.to(torch.float32)[is_labeled],
                             batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))
                else:
                    loss = mse_reg_criterion(pred.to(torch.float32)[is_labeled],
                                             batch.y.to(torch.float32)[is_labeled].view(pred[is_labeled].size()))

            loss_curve.append(loss.detach().cpu().item())
            total_loss += loss.item() * batch.num_graphs
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), total_loss / len(loader.dataset), sum(loss_curve)/len(loss_curve), y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--patience', default=200, type=int)
    parser.add_argument('--dataset', type=str, default="Freesolv")
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--eval_metric', type=str, default="rocauc")
    parser.add_argument('--task_num', type=int, default=1)
    parser.add_argument('--l_max', type=int, default=2)
    parser.add_argument('--num_radial', type=float, default=6)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hire_layers', type=int, default=1)
    parser.add_argument('--cutoff', type=float, default=3.5) 
    parser.add_argument('--L', type=int, default=2)
    parser.add_argument('--Nrad', type=int, default=8)
    parser.add_argument('--mul_l0', type=int, default=32)
    parser.add_argument('--mul_l1', type=int, default=16)
    parser.add_argument('--mul_l2', type=int, default=8)
    parser.add_argument('--so3_step', type=float, default=0.2)
    parser.add_argument('--num_z_embeddings', type=int, default=9)
    parser.add_argument('--target', type=str, default='aaa')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--seeds', type=int, default=4)

    args = parser.parse_args()

    if args.dataset in ["Esol", "Freesolv", "Lipop"]:
        args.eval_metric = "rmse"
    elif args.dataset == "qm9":
        args.eval_metric = "mae"  

    torch.manual_seed(3047)
    np.random.seed(3047)

    if os.getcwd()[-10:] != 'benchmarks':
        os.chdir(os.getcwd())

    if not os.path.exists(os.path.join(os.getcwd(), 'results', args.dataset)):
        os.makedirs(os.path.join(os.getcwd(), 'results', args.dataset))
    if not os.path.exists(os.path.join(os.getcwd(), 'logs', args.dataset)):
        os.makedirs(os.path.join(os.getcwd(), 'logs', args.dataset))
    if not os.path.exists(os.path.join('models', args.dataset)):
        os.makedirs(os.path.join('models', args.dataset))
    now = "_" + "-".join(str(datetime.datetime.today()).split()).split('.')[0].replace(':', '.')
    weight_decay = "wd_" + str(args.weight_decay) if args.weight_decay > 0 else ""
    program_name = f'_bs_{args.batch_size}' + '_lr_' + str(
        args.lr) + weight_decay + now

    logging.basicConfig(filename=os.path.join('logs', args.dataset, program_name + '.log'), level=logging.INFO,
                        filemode="w")
    log = PrinterLogger(logging.getLogger(__name__))
    print(args)

    test_perfs_at_best, test_perfs_at_end, test_losses = [], [], []
    val_perfs_at_best, val_perfs_at_end, val_losses = [], [], []
    train_perfs_at_end, train_losses = [], []
    best_epochs = []
    
    dataset = get_dataset(args.dataset, args.output_dir).to(device)

    print(f"\n{'='*20} Pre-Check {'='*20}")
    if hasattr(dataset[0], 'geo_cos_phi'):
        print("✅ Geometric features found.")
    else:
        print("❌ Warning: Geometric features missing! Check preprocess.")

    if args.dataset in ["BBBP", "Bace", "ClinTox", "Sider", "Tox21", "HIV", "BBBP_cutoff"]:
        maximize = True
    else:
        maximize = False

    if args.dataset in ["BBBP", "ClinTox", "Tox21", "Sider", "HIV",
                        "Esol", "Freesolv", "Lipop", "Bace", "qm9", "BBBP_cutoff"]:
        dataset.data.y = dataset.data.y.masked_fill(torch.isnan(dataset.data.y), 0)
        
        split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=math.floor(dataset.data.y.shape[0] * 0.8),
                                          valid_size=math.floor(dataset.data.y.shape[0] * 0.1), seed=42)
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=args.batch_size, shuffle=False)

    mean, std = None, None
    atom_refs = None
    
    if args.dataset in ["qm9", "Esol", "Freesolv", "Lipop"]:
        print(f"Processing statistics for {args.dataset}...")
        
        train_dataset = dataset[split_idx['train']]
        y_train = train_dataset.y
        
        mean = y_train.mean(dim=0, keepdim=True)
        std = y_train.std(dim=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)

        if args.dataset == "qm9":
            print("Calculating AtomRefs for QM9 (Robust Least Squares)...")
            
            max_z = 100
            atom_counts_list = []
            target_values_list = []
            
            for i in tqdm(range(len(train_dataset)), desc="Gathering Atom Stats"):
                data = train_dataset[i]
                z = data.x[:, 0].long()
                counts = torch.bincount(z, minlength=max_z)
                atom_counts_list.append(counts)
                target_values_list.append(data.y.view(-1)[0].item())
            
            A = torch.stack(atom_counts_list).float().to(device)
            y_targets = torch.tensor(target_values_list).float().to(device)
            
            col_sum = A.sum(dim=0)
            present_atom_indices = torch.nonzero(col_sum).squeeze()
            
            print(f"Atoms present in dataset (Z): {present_atom_indices.tolist()}")
            
            A_sub = A[:, present_atom_indices]
            
            try:
                result = torch.linalg.lstsq(A_sub, y_targets, rcond=None)
                coeffs_sub = result.solution
            except Exception as e:
                print(f"Lstsq failed: {e}, using pseudoinverse instead.")
                coeffs_sub = torch.matmul(torch.linalg.pinv(A_sub), y_targets)
            
            ref_solution = torch.zeros(max_z, device=device)
            ref_solution[present_atom_indices] = coeffs_sub
            
            atom_refs = ref_solution.cpu()
            
            print("Calculated AtomRefs (subset):")
            for idx in present_atom_indices:
                print(f"  Z={idx.item()}: {atom_refs[idx].item():.4f}")

            if torch.isnan(atom_refs).any():
                raise ValueError("❌ AtomRefs calculation resulted in NaNs! Check data or solver.")
            print("✅ AtomRefs calculated successfully.")

    evaluator = Evaluator(args.task_num, args.eval_metric)

    all_seed_test_preds = [] 
    test_y_true = None 

    n_seeds = args.seeds
    custom_seeds = [3047, 101, 202, 303][:n_seeds]
    
    log.print_and_log("\n" + "-" * 15 + f" TESTING OVER {n_seeds} SEEDS: {custom_seeds}" + "-" * 15 + "\n")

    model = ProductManifoldGNN(L=args.L, Nrad=args.Nrad, r_cut=args.cutoff, n_layers=args.layers,
                               n_hire_layers=args.hire_layers, mul_l0=args.mul_l0, mul_l1=args.mul_l1, mul_l2=args.mul_l2,
                               so3_step=args.so3_step, num_z_embeddings=args.num_z_embeddings, hidden_dim=args.hidden_dim,
                               task_num=args.task_num,atom_refs=atom_refs,target_name=args.target)
    model = model.to(device)

    if mean is not None and std is not None:
        model.register_buffer("mean", mean)
        model.register_buffer("std", std)
        print("✅ Mean and Std injected into model.")

    print("Initializing EMA model with decay 0.999...")
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: \
            0.999 * averaged_model_parameter + 0.001 * model_parameter
    ema_model = AveragedModel(model, device=device, avg_fn=ema_avg)
    
    if hasattr(model, 'mean'):
        ema_model.mean = model.mean
    if hasattr(model, 'std'):
        ema_model.std = model.std

    fs = []
    log.print_and_log(f'Model # Parameters {sum([p.numel() for p in model.parameters()])}')
    
    for i, seed in enumerate(custom_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        model.reset_parameters()
        if args.weight_decay == 0:
            optim_class = optim.Adam
        else:
            optim_class = optim.AdamW

        optimizer = optim_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=10)
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10])

        model_seed_save_path = os.path.join('models', args.dataset, f'model_best_{program_name}_seed_{seed}.pth.tar')
        early_stopper = Patience(patience=args.patience, use_loss=False, save_path=model_seed_save_path, maximize=maximize)

        best_test_pred_this_seed = None

        pbar = tqdm(range(args.epochs), desc=f"{i+1}/{n_seeds} (Seed {seed})")
        for epoch in pbar:
            train_loss_avg = train(model, device, train_loader, optimizer, dataset.task_type, args.dataset, ema_model=ema_model)
            train_losses.append(train_loss_avg)
            
            valid_perf, val_loss, val_loss_curve, _, _ = eval(ema_model, device, valid_loader, evaluator, dataset.task_type, args.dataset)
            
            test_perf, test_loss, test_loss_curve, t_y_true, t_y_pred = eval(ema_model, device, test_loader, evaluator, dataset.task_type, args.dataset)
            
            val_losses.append(val_loss_curve)
            test_losses.append(test_loss_curve)

            if maximize:
                is_best = valid_perf[args.eval_metric] >= early_stopper.val_acc
            else:
                is_best = valid_perf[args.eval_metric] <= early_stopper.val_acc
            
            if is_best:
                test_perf_at_best = test_perf
                best_test_pred_this_seed = t_y_pred
                if test_y_true is None:
                    test_y_true = t_y_true 

            if scheduler is not None:
                scheduler.step()

            if early_stopper.stop(epoch, val_loss, valid_perf[args.eval_metric], model=ema_model):
                break

            pbar.set_description(
                f"Seed {seed} | L:{train_loss_avg:.4f} | Val (EMA): {valid_perf[args.eval_metric]:.3f} | Best: {early_stopper.val_acc:.3f} | Test (EMA): {test_perf[args.eval_metric]:.3f}")

        test_perf_at_end, _, _, _, _ = eval(model, device, test_loader, evaluator, dataset.task_type, args.dataset)
        train_perf_at_end, _, _, _, _ = eval(model, device, train_loader, evaluator, dataset.task_type, args.dataset)

        train_perfs_at_end.append(train_perf_at_end[args.eval_metric])
        val_perfs_at_end.append(valid_perf[args.eval_metric])
        
        val_perfs_at_best.append(early_stopper.val_acc)
        
        checkpoint = torch.load(model_seed_save_path)
        
        ema_model.load_state_dict(checkpoint['state_dict'])
        
        final_test_perf, _, _, _, final_test_y_pred = eval(ema_model, device, test_loader, evaluator, dataset.task_type, args.dataset)
        
        test_perfs_at_best.append(final_test_perf[args.eval_metric])
        best_epochs.append(early_stopper.best_epoch + 1)

        msg = (
            f'============= Results Seed {seed} =============\n'
            f'Dataset:        {args.dataset}\n'
            f'Best epoch:     {early_stopper.best_epoch + 1}\n'
            f'Validation:     {early_stopper.val_acc:0.4f}\n'
            f'Test (Best):    {final_test_perf[args.eval_metric]:0.4f}\n'
            '-------------------------------------------\n\n')

        log.print_and_log(msg)
        fs.append(final_test_perf[args.eval_metric])


    log.print_and_log("\n" + "=" * 15 + " FINAL RESULTS (NO ENSEMBLE) " + "=" * 15)
    log.print_and_log(f"Test Perf (Avg over seeds): {np.mean(test_perfs_at_best):.4f} +/- {np.std(test_perfs_at_best):.4f}")
    log.print_and_log("=" * 48 + "\n")


    results = {
        "perfs_at_end": {"train": np.asarray(train_perfs_at_end), "val": np.asarray(val_perfs_at_end),
                         "test": np.asarray(test_perfs_at_best)},
        "perfs_at_best": {"val": np.asarray(val_perfs_at_best), "test": np.asarray(test_perfs_at_best)},
        "best_epochs": np.asarray(best_epochs),
        "train_loss_curve": np.asarray(train_losses),
        "val_loss_curve": np.asarray(val_losses),
        "test_loss_curve": np.asarray(test_losses)}

    with open(os.path.join("results", args.dataset, args.dataset + "_" + "results_" + program_name + ".pkl"),
              "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True) 
    main()