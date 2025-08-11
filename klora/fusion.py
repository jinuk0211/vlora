from typing import Optional, Union
import torch
from torch import nn
glo_count = 0
import numpy as np
class KLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_1_a: torch.Tensor,
        weight_1_b: torch.Tensor,
        weight_2_a: torch.Tensor,
        weight_2_b: torch.Tensor,
        average_ratio: float = 1.0,
        rank: int = 8,
        alpha: int = 1.5,
        beta: int = 0.5,
        sum_timesteps: int = 28000,
        pattern:str = "s*",
        device: Optional[Union[torch.device, str]] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.device = device
        self.weight_1_a = weight_1_a.to(device)
        self.weight_1_b = weight_1_b.to(device)
        self.weight_2_a = weight_2_a.to(device)
        self.weight_2_b = weight_2_b.to(device)
        self.average_ratio = average_ratio
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.sum_timesteps = sum_timesteps
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"
        self.pattern = pattern
        
    def get_weights(self, timestep, threshold =1):
        content_matrix = self.weight_1_a @ self.weight_1_b
        style_matrix = self.weight_2_a @ self.weight_2_b 

        style_norm = torch.norm(style_matrix, p='fro')
        content_norm = torch.norm(content_matrix, p='fro')

        # print("Style norm:", style_norm.item())
        # print("Content norm:", content_norm.item())

        alpha = (style_norm.item() + content_norm.item()) / 2
        # print("Alpha:", alpha)

        style_lora_normed = (alpha / style_norm.item()) * style_matrix
        content_lora_normed = (alpha / content_norm.item()) * content_matrix
        style_lora_normed = style_lora_normed.to(torch.float32)
        content_lora_normed = content_lora_normed.to(torch.float32)
        # S_s= torch.linalg.svdvals(style_lora_normed)
        # S_c= torch.linalg.svdvals(content_lora_normed)
        U_s, S_s, V_s = torch.svd_lowrank(style_lora_normed,q =10)
        U_c, S_c, V_c = torch.svd_lowrank(content_lora_normed,q=10)        
        style_score = S_s.sum()
        content_score = S_c.sum()
        # print(f'style_score:{style_score}')
        # print(f'content_score:{content_score}')
        # if content_score - style_score > 0:
        #     print('content matrix win')
        # else:
        #     print('style matrix win')
        gamma = 1000 # threshold 0~1
        S = (gamma - 1) * np.exp(-60 * timestep / self.sum_timesteps) + 1     
        v = torch.abs(content_score - style_score)
        content_score = content_score * S
        print(f'v:{v}')        
        if v > threshold:
            if content_score > style_score:
                print(f'content matrix win: {v}')
                return content_matrix
            else:
                print(f'style matrix win: {v}')
                return style_matrix
        else:
            # if timestep / self.sum_timesteps > 0.9:
            #     print(f'no matrix win t > 0.9 :{v}')
            #     a = 0.001
            #     x = timestep - 0.9 * self.sum_timesteps
            #     lamb = 1 + (1 - np.exp(-a * x))
            #     return content_matrix + lamb * style_matrix
            if timestep / self.sum_timesteps > 0.8:
                a = 0.001
                x = timestep - 0.8 * self.sum_timesteps
                lamb = 1 + (1 - np.exp(-a * x))
                style_score = style_score * lamb
            if style_score > content_score and timestep / self.sum_timesteps > 0.8:
                print(f'no matrix win t = {timestep/self.sum_timesteps} - high score style matrix return :{v}')
                return style_matrix
            print(f'no matrix win:{v}')
            return content_matrix + style_matrix
        # alpha = style_score / (style_score + content_score + 1e-8)
        # combined_matrix = alpha * style_matrix + (1 - alpha) * content_matrix

        #factor 만들기


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count # 클래스의 인스턴스 변수x, 모듈 전역 변수를 참조 사실상 timestep
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            # weight = self.get_weights(glo_count)
            glo_count += 1
            weight = self.weight_1_a @ self.weight_1_b + self.weight_2_a @ self.weight_2_b #content, style lora 그냥 합치기
        elif self.forward_type == 'style':
            weight = self.weight_2_a @ self.weight_2_b 
        elif self.forward_type == 'content':
            weight = self.weight_1_a @ self.weight_1_b
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class KLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)

        # style_norm = torch.norm(style_lora)
        # content_norm = torch.norm(content_lora)

        # alpha = (style_norm + content_norm) / 2

        # style_lora_normed = (alpha / style_norm) * style_lora
        # content_lora_normed = (alpha / content_norm) * content_lora
        # U_s, S_s, V_s = torch.svd(style_lora_normed)
        # U_c, S_c, V_c = torch.svd(content_lora_normed)
        # style_score = S_s.sum()
        # content_score = S_c.sum()
        # v = torch.abs(content_score - style_score)
