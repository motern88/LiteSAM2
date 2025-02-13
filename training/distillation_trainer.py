import torch  
from torch import nn  
from trainer import Trainer  # 继承原始训练器  
from typing import Any, Dict, List, Optional
from torch.nn import functional as F

class StageAwareDistillationTrainer(Trainer):  
    """支持分阶段蒸馏的扩展训练器"""  
    
    def __init__(  
        self,  
        # 新增参数  
        teacher: nn.Module,  
        student: nn.Module,  
        # 原始参数继承  
        *,  
        data: Dict[str, Any],  
        model: Dict[str, Any],  
        logging: Dict[str, Any],  
        checkpoint: Dict[str, Any],  
        max_epochs: int,  
        mode: str = "train",  
        accelerator: str = "cuda",  
        seed_value: int = 123,  
        val_epoch_freq: int = 1,  
        distributed: Dict[str, bool] = None,  
        cuda: Dict[str, bool] = None,  
        env_variables: Optional[Dict[str, Any]] = None,  
        optim: Optional[Dict[str, Any]] = None,  
        optim_overrides: Optional[List[Dict[str, Any]]] = None,  
        meters: Optional[Dict[str, Any]] = None,  
        loss: Optional[Dict[str, Any]] = None,  
        # 蒸馏特有参数  
        temperature: float = 3.0,  
        alpha: float = 0.7,  
        feature_layers: List[int] = [4, 8, 12]  
    ):  
        # 初始化父类（原始训练器）  
        super().__init__(  
            data=data,  
            model=model,  
            logging=logging,  
            checkpoint=checkpoint,  
            max_epochs=max_epochs,  
            mode=mode,  
            accelerator=accelerator,  
            seed_value=seed_value,  
            val_epoch_freq=val_epoch_freq,  
            distributed=distributed,  
            cuda=cuda,  
            env_variables=env_variables,  
            optim=optim,  
            optim_overrides=optim_overrides,  
            meters=meters,  
            loss=loss  
        )  
        
        # 蒸馏特有初始化  
        self.teacher = teacher.to(self.device)  
        self.student = student.to(self.device)  
        self.temperature = temperature  
        self.alpha = alpha  
        self.feature_layers = feature_layers  
        
        # 冻结教师模型  
        for param in self.teacher.parameters():  
            param.requires_grad_(False)  
            
        # 替换原始模型为学生模型  
        self.model = self.student  # 关键：将父类的model替换为学生模型  

    def _compute_distill_loss(self, teacher_out, student_out):  
        """计算蒸馏损失"""  
        # 特征对齐损失  
        feat_loss = sum([  
            F.mse_loss(teacher_out.features[i], student_out.features[i])  
            for i in self.feature_layers  
        ])  
        
        # KL散度损失  
        kl_loss = nn.KLDivLoss(reduction='batchmean')(  
            F.log_softmax(student_out.logits / self.temperature, dim=1),  
            F.softmax(teacher_out.logits / self.temperature, dim=1)  
        ) * (self.temperature ** 2)  
        
        return self.alpha * kl_loss + (1 - self.alpha) * feat_loss  

    def training_step(self, batch, batch_idx):  
        """重写训练步骤"""  
        # 教师模型前向（不计算梯度）  
        with torch.no_grad():  
            teacher_out = self.teacher(batch["images"])  
            
        # 学生模型前向  
        student_out = self.student(batch["images"])  
        
        # 计算原始任务损失  
        task_loss = super().training_step(batch, batch_idx)  
        
        # 计算蒸馏损失  
        distill_loss = self._compute_distill_loss(teacher_out, student_out)  
        
        # 合并损失  
        total_loss = task_loss + distill_loss  
        
        # 记录指标  
        self.logger.log("Losses/distill", distill_loss, self.steps["train"])  
        self.logger.log("Losses/total", total_loss, self.steps["train"])  
        
        return total_loss  

    def configure_optimizers(self):  
        """分阶段优化器配置"""  
        # 从父类获取基础优化器  
        optimizer = super().configure_optimizers()  
        
        # 添加分阶段参数组  
        optimizer.param_groups.extend([  
            # Phase 2: 解冻memory_attention  
            {  
                "params": self.student.memory_attention.parameters(),  
                "lr": 1e-4,  
                "phase": 2  
            },  
            # Phase 3: 解冻sam_mask_decoder  
            {  
                "params": self.student.sam_mask_decoder.parameters(),  
                "lr": 2e-4,  
                "phase": 3  
            }  
        ])  
        return optimizer  

    # 通过 on_epoch_start 回调控制参数组激活状态
    def on_epoch_start(self, epoch):  
        """阶段感知优化器更新"""  
        current_phase = self._get_current_phase(epoch)  
        
        # 动态启用/禁用参数组  
        for group in self.optim.optimizer.param_groups:  
            if "phase" in group:  
                group["active"] = (group["phase"] <= current_phase)  
                
        super().on_epoch_start(epoch)