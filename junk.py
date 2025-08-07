    if content_layer_nums == 190:
        unet = unet.transformer
        for attn_processor_name, attn_processor in unet.attn_processors.items():
            attn_module = unet
            for n in attn_processor_name.split(".")[:-1]:
                attn_module = getattr(attn_module, n)
            attn_name = ".".join(attn_processor_name.split(".")[:-1])
            merged_lora_weights_dict_1_a, merged_lora_weights_dict_1_b = (
                merge_community_flux_lora_weights(  #    tensors: torch.Tensor, key: str, prefix: str = "transformer.", layer_num: int = 0
                    tensors=lora_weights_content,
                    key=attn_name,
                    layer_num=content_layer_nums,
                )
            )
# target_key = prefix + key + "." down_key = target_key + f"{part}.lora_A.weight" up_key = target_key + f"{part}.lora_B.weight" out1[part] = tensors[up_key]
            merged_lora_weights_dict_2_a, merged_lora_weights_dict_2_b = (
                merge_community_flux_lora_weights(
                    tensors=lora_weights_style,
                    key=attn_name,
                    layer_num=style_layer_nums,
                )
            )

            kwargs = {
                "alpha": alpha,
                "beta": beta,
                "sum_timesteps": sum_timesteps,
                "average_ratio": average_ratio,
                "patten": patten,
                "state_dict_1_a": merged_lora_weights_dict_1_a,
                "state_dict_1_b": merged_lora_weights_dict_1_b,
                "state_dict_2_a": merged_lora_weights_dict_2_a,
                "state_dict_2_b": merged_lora_weights_dict_2_b,
            }
            # Set the `lora_layer` attribute of the attention-related matrices.
            copy_and_assign_klora_weights(attn_module, "to_q")
            copy_and_assign_klora_weights(attn_module, "to_k")
            copy_and_assign_klora_weights(attn_module, "to_v")

            to_k = LoRACompatibleLinear(
                in_features=attn_module.to_k.in_features,
                out_features=attn_module.to_k.out_features,
                bias=True,
                device=attn_module.to_k.weight.device,
                dtype=attn_module.to_k.weight.dtype,
            )
            to_k.weight.data = attn_module.to_k.weight.data.clone()
            to_k.bias.data = attn_module.to_k.bias.data.clone()
            attn_module.to_k = to_k

            attn_module.to_q.set_lora_layer(
                initialize_klora_layer(
                    **kwargs,
                    part="to_q",
                    in_features=attn_module.to_q.in_features,
                    out_features=attn_module.to_q.out_features,
                )
            )
            attn_module.to_k.set_lora_layer(
                initialize_klora_layer(
                    **kwargs,
                    part="to_k",
                    in_features=attn_module.to_k.in_features,
                    out_features=attn_module.to_k.out_features,
                )
            )
            attn_module.to_v.set_lora_layer(
                initialize_klora_layer(
                    **kwargs,
                    part="to_v",
                    in_features=attn_module.to_v.in_features,
                    out_features=attn_module.to_v.out_features,
                )
            )

            if not ("single" in attn_name):
                attn_module.to_out[0].set_lora_layer(
                    initialize_klora_layer(
                        **kwargs,
                        part="to_out.0",
                        in_features=attn_module.to_out[0].in_features,
                        out_features=attn_module.to_out[0].out_features,
                    )
                )