from typing import Optional, Union
import torch
from torch import nn
glo_count = 0

class SingularLoRALinearLayer(nn.Module):
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

    def get_weights(self, timestep, threshold =0.7):
        content = self.weight_1_a @ self.weight_1_b
        style = self.weight_2_a @ self.weight_2_b 

        # U_s, S_s, V_s = torch.svd(style_matrix)   # style
        # U_c, S_c, V_c = torch.svd(content_matrix) # content
        # k = 4  # 원하는 rank
        # style_score = S_s[:k].sum()
        # content_score = S_c[:k].sum()
        # style_ratio = S_s[:k].sum() / S_s.sum()
        # content_ratio = S_c[:k].sum() / S_c.sum()

        #frobenius norm 근사        #frobenius norm 근사
        # style_frob = torch.norm(S_s, p='fro')  # == torch.sqrt((S_s**2).sum())
        # content_frob = torch.norm(S_c, p='fro')

        style_norm = torch.norm(style_lora)
        content_norm = torch.norm(content_lora)

        # Step 2: Compute α (average norm)
        alpha = (style_norm + content_norm) / 2

        # Step 3: Normalize each LoRA
        style_lora_normed = (alpha / style_norm) * style_lora
        content_lora_normed = (alpha / content_norm) * content_lora
        U_s, S_s, V_s = torch.svd(style_lora_normed)
        U_c, S_c, V_c = torch.svd(content_lora_normed)
        style_score = S_s.sum()
        content_score = S_c.sum()

        v = torch.abs(content_score - style_score)
        if v > threshold:
            if content_score > style_score:
            print(f'content matrix win: {v}')
            return content_matrix
            else:
            print(f'style matrix win: {v}')
            return style_matrix
        else:
            content_matrix + style_matrix
        # alpha = style_score / (style_score + content_score + 1e-8)
        # combined_matrix = alpha * style_matrix + (1 - alpha) * content_matrix

        #factor 만들기


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        global glo_count # 클래스의 인스턴스 변수x, 모듈 전역 변수를 참조 사실상 timestep
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1_a.dtype

        if self.forward_type == "merge":
            weight = self.get_weights(glo_count)
            glo_count += 1
            # weight = self.weight_1_a @ self.weight_1_b + self.lambda * self.weight_2_a @ self.weight_2_b #content, style lora 그냥 합치기
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