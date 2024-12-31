import argparse
import random
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sml import (
    build_train_and_test_dataset,
    FeedForwardNetwork,
    LiuModel,
    LookUpTable,
    HybridModel,
    set_seed,
    get_logger,
    save_checkpoint
)

logger = get_logger(__name__)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, 
                       choices=["ff", "liu", "lut", "hybrid-w-liu", "hybrid-w-lut"], 
                       required=True)
    parser.add_argument("--ffn_layers", type=str, nargs="+", default=[8, 50, 50, 50, 1])
    parser.add_argument("--data_paths", 
                        type=lambda x: [path.strip() for path in x.split(",")],
                        required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--save_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--lr_patience", type=int, default=200)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    # parser.add_argument("--kfold", type=int, default=5)

    return parser.parse_args()


def evaluate(model, test_dataset, device, criterion, args):
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model.eval()
    test_loss = 0
    all_errors = []
    all_preds = []
    all_targets = []
    all_data = []
    kwargs = {"dtype": torch.float32, "device": device}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(**kwargs), target.to(**kwargs)
            output = model(data)
            # logger.info(f"target.shape: {target.shape}, output.shape: {output.shape}")
            
            # pred = output.squeeze()
            if output.dim() == 0:  # 如果是标量，转换为1维张量
                output = output.unsqueeze(0)
            if target.dim() == 0:
                target = target.unsqueeze(0)
            output = output.view(-1)
            target = target.view(-1)

            abs_rel_errors = torch.abs(output - target) / target
            all_data.extend(data.cpu().numpy().tolist())
            all_errors.extend(abs_rel_errors.cpu().numpy().tolist())
            all_preds.extend(output.cpu().numpy().tolist())
            all_targets.extend(target.cpu().numpy().tolist())

            test_loss += criterion(output.squeeze(), target).item()
    
    all_errors = sorted(all_errors)
    total_samples = len(all_errors)
    cumulative_fractions = [(i + 1) / total_samples for i in range(total_samples)]
    
    avg_loss = test_loss / len(test_loader)
    
    # 计算rRMSE和RMSE
    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2))
    rrmse = torch.sqrt(torch.mean(((all_preds - all_targets) / all_targets) ** 2))
    # logger.info(f"RMSE: {rmse:.4f}")
    # logger.info(f"rRMSE: {rrmse:.4f}")

    # 给10个sample, 显示pred和target
    sample_indices = random.sample(range(len(all_preds)), 10)
    for idx in sample_indices:
        logger.info(f"Sample {idx+1}: Data = {all_data[idx]}, Pred = {all_preds[idx]:.4f}, Target = {all_targets[idx]:.4f}")
    
    # error_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    error_thresholds = [0.01, 0.1, 0.5]
    for threshold in error_thresholds:
        idx = next((i for i, err in enumerate(all_errors) if err > threshold), len(all_errors))
        fraction = cumulative_fractions[idx-1] if idx > 0 else 0.0
        logger.info(f"Cumulative fraction at [relative error = {threshold*100:.0f}%] is {fraction*100:.1f}%")
    
    return avg_loss, (all_errors, cumulative_fractions)


def train_epoch(model, train_loader, device, criterion, optimizer):
    model.train()
    total_loss = 0
    kwargs = {"dtype": torch.float32, "device": device}
    
    cumualtive_mse_loss = 0
    train_loader = tqdm(train_loader, desc="Training", leave=False)
    for data, target in train_loader:
        data = data.to(**kwargs)
        target = target.to(**kwargs)
        
        
        optimizer.zero_grad()
        output = model(data).squeeze()
        if output.dim() == 0:  # 如果是标量，转换为1维张量
            output = output.unsqueeze(0)
        if target.dim() == 0:  # 如果是标量，转换为1维张量
            target = target.unsqueeze(0)
        output = output.view(-1)
        target = target.view(-1)
        # logger.info(f"data: {data}, target: {target}, output: {output}, len(dataloader): {len(train_loader)}")
        
        # manually calculate RMSE, rRMSE
        mse = torch.mean((output - target) ** 2)
        rmse = torch.sqrt(torch.mean((output - target) ** 2))
        rrmse = torch.sqrt(torch.mean(((output - target) / target) ** 2))
        cumualtive_mse_loss += mse.item()
        # logger.info(f"RMSE: {rmse:.4f}, rRMSE: {rrmse:.4f}, cumualtive_mse_loss: {cumualtive_mse_loss:.4f}")
        # time.sleep(1)

        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        # logger.info(f"output.shape: {output.shape}, target.shape: {target.shape}, loss: {loss}, total_loss: {total_loss}")
    
    # logger.info(f"cumualtive_mse_loss: {cumualtive_mse_loss:.4f}, avg_mse_loss: {cumualtive_mse_loss / len(train_loader):.4f}")
    # logger.info(f"Train loss: total_loss: {total_loss:.4f}, len(train_loader): {len(train_loader)}, avg_loss: {total_loss / len(train_loader):.4f}")
    return total_loss / len(train_loader)


def train(model, train_dataset, test_dataset, device, criterion, optimizer, args):
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists() and not checkpoint_dir.is_dir():
        checkpoint_dir.unlink()  # 如果是文件则删除
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    best_loss = float('inf')
    # patience_counter = 0  # 用于追踪验证损失未改善的轮数
    # last_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, device, criterion, optimizer)
        # time.sleep(15)
        
        if (epoch + 1) % args.save_epochs == 0:
            test_loss, (all_errors, cumulative_fractions) = evaluate(model, test_dataset, device, criterion, args)
            
            # # 学习率衰减逻辑
            # current_lr = optimizer.param_groups[0]['lr']
            # if epoch > args.lr_patience:
            #     new_lr = max(current_lr * args.lr_decay, args.min_lr)
            
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = new_lr
            
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} - "
                f"Train loss: {train_loss:.4f}, "
                f"Test loss: {test_loss:.4f}, "
                f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            if test_loss < best_loss:
                best_loss = test_loss
                save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_dir / "best_model.pt")
                
    
    return best_loss


def main(args):

    logger.info(f"args: {args}")

    set_seed(args.seed)

    train_dataset, test_dataset = build_train_and_test_dataset(args.data_paths)

    logger.info(f"train_dataset: {len(train_dataset)}, test_dataset: {len(test_dataset)}")

    # only use 100 samples for training
    # indices = list(range(len(train_dataset)))[:1000]
    # train_dataset = torch.utils.data.Subset(train_dataset, indices)
    # indices = list(range(len(test_dataset)))[:100]
    # test_dataset = torch.utils.data.Subset(test_dataset, indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    
    logger.info(
        "\n========================================================"
        "Train!"
        "========================================================\n"
    )
    if args.model in ["liu", "lut"]:
        train_dataset, test_dataset = build_train_and_test_dataset(args.data_paths, only_physical_features=True)
        # logger.info(f"train_dataset: {train_dataset}")
        # # print(train_dataset.shape)
        # logger.info(f"train_dataset[0]: {train_dataset[0]}")
        # train_dataset = train_dataset[:, 3:]
        # test_dataset = test_dataset[:, 3:]
        if args.model == "liu":
            model = LiuModel()
        else:
            model = LookUpTable()
        
        model = model.to(device)
        optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate)
        best_val_loss = train(model, train_dataset, test_dataset, device, criterion, optimizer, args)
        # test_loss, _ = evaluate(model, test_dataset, device, criterion, args)
        # logger.info(f"Test loss: {test_loss:.4f}")
    
    else:
        if args.model == "ff":
            model = FeedForwardNetwork()
        elif args.model == "hybrid-w-liu":
            model = HybridModel(physical_model="liu")
        elif args.model == "hybrid-w-lut":
            model = HybridModel(physical_model="lut")
            
        model = model.to(device)
        optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate)

        best_val_loss = train(model, train_dataset, test_dataset, device, criterion, optimizer, args)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        # 加载最佳模型时确保文件路径正确
        best_model_path = Path(args.checkpoint_dir) / "best_model.pt"
        if best_model_path.is_file():  # 确保文件存在且不是目录
            ckpt = torch.load(best_model_path)
            model.load_state_dict(ckpt["model_state_dict"])
            test_loss, _ = evaluate(model, test_dataset, device, criterion, args)
            logger.info(f"最终测试损失: {test_loss:.4f}")
        else:
            logger.error(f"找不到最佳模型文件: {best_model_path}")



if __name__ == "__main__":
    args = parse_args()
    main(args)