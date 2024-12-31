from pathlib import Path
from tqdm import tqdm

import torch

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        checkpoint_dir="./checkpoints",
        batch_size=32,
        num_workers=4
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="训练") as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return total_loss / len(train_loader)
    

    
    def train(self, train_lo, val_dataset, epochs=100):
        
        
        for epoch in range(epochs):
            print(f"\n轮次 {epoch+1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(f"best_model.pt", epoch, val_loss)
                
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, val_loss)
    
