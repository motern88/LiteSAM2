from torch import nn

# 教师模型包装类  
class TeacherWrapper(nn.Module):  
    def __init__(self, teacher_model):  
        super().__init__()  
        self.teacher = teacher_model  
        for param in self.teacher.parameters():  
            param.requires_grad_(False)  
            
        self.feature_maps = {}  
        self._register_hooks()  
    
    def _register_hooks(self):  
        # 指定需要捕获的模块  
        target_modules = {  
            "image_encoder": self.teacher.image_encoder,  
            "memory_attention": self.teacher.memory_attention,  
            "sam_mask_decoder": self.teacher.sam_mask_decoder  
        }  
        for name, module in target_modules.items():  
            module.register_forward_hook(  
                self._make_hook(name)  
            )  
    
    def _make_hook(self, name):  
        def hook(module, input, output):  
            self.feature_maps[name] = output.detach()  
        return hook  

    def forward(self, x):  
        self.feature_maps.clear()  
        return self.teacher(x)