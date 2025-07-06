from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from sglang.srt.distributed import divide
from sglang.srt.hf_transformers_utils import AutoConfig
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
    VOCAB_PARALLELISM_EMBEDDING_NAMES,
    ROW_PARALLELISM_LINEAR_LORA_NAMES,
    LoRAType,
    get_hidden_dim,
    get_stacked_multiply,
    get_weight_name,
)


class LoRAMemoryPool:
    """Class for memory pool management of lora modules"""

    def __init__(
        self,
        base_hf_config: AutoConfig,
        max_loras_per_batch: int,
        dtype: torch.dtype,
        tp_size: int,
        tp_rank: int,
        max_extra_vocab_size: int,
    ):
        self.base_hf_config: AutoConfig = base_hf_config
        self.num_layer: int = base_hf_config.num_hidden_layers
        self.max_loras_per_batch: int = max_loras_per_batch
        self.dtype: torch.dtype = dtype
        self.tp_size: int = tp_size
        self.tp_rank: int = tp_rank
        self.max_extra_vocab_size: int = max_extra_vocab_size

        # Both A_buffer and B_buffer maps lora weight names to its buffer space.
        # A_buffer contains num_layer number of row-major tensors with shape
        #   (max_loras_per_batch, stacked_num * max_lora_dim, input_dim)
        # B_buffer contains num_layer number of column-major tensors with shape
        #   (stacked_num, max_loras_per_batch, output_dim, max_lora_dim)
        self.A_buffer: Dict[str, List[torch.Tensor]] = {}
        self.B_buffer: Dict[str, List[torch.Tensor]] = {}

        self.new_embeddings_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_A_buffer: Dict[str, torch.Tensor] = {}
        self.embedding_B_buffer: Dict[str, torch.Tensor] = {}

        self.embedding_dim: int = self.base_hf_config.hidden_size

        # Lora uid -> buffer idx in memory pool
        self.uid_to_buffer_id: Dict[Optional[str], int] = {}

        # Buffer idx -> lora uid in memory pool
        # All uids are initialized as empty strings for empty buffer slots
        # Here we don't initialize to None since None is a valid uid
        self.buffer_id_to_uid: List[Optional[str]] = [""] * self.max_loras_per_batch

    def get_lora_A_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        input_dim, _ = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            input_dim = divide(input_dim, self.tp_size)
        return (
            self.max_loras_per_batch,
            max_lora_dim * c,
            input_dim,
        )

    def get_lora_B_shape(
        self, module_name: str, base_model: torch.nn.Module, max_lora_dim: int
    ) -> Tuple[int]:
        """
        Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
        """
        _, output_dim = get_hidden_dim(module_name, self.base_hf_config, base_model)
        c = get_stacked_multiply(module_name)
        if self.tp_size > 1 and module_name not in ROW_PARALLELISM_LINEAR_LORA_NAMES:
            output_dim = divide(output_dim, self.tp_size)
        return (
            c,
            self.max_loras_per_batch,
            output_dim,
            max_lora_dim,
        )

    def get_embedding_lora_A_shape(
        self, max_lora_dim: int
    ) -> Tuple[int]:
        base_vocab_size = self.base_hf_config.vocab_size
        max_extra_vocab_size = base_vocab_size + self.max_extra_vocab_size
        return (
            self.max_loras_per_batch,
            max_lora_dim,
            max_extra_vocab_size,
        )

    def get_embedding_lora_B_shape(
        self, max_lora_dim: int
    ) -> Tuple[int]:
        embedding_dim = self.base_hf_config.hidden_size
        return (
            self.max_loras_per_batch,
            embedding_dim,
            max_lora_dim,
        )

    def init_buffers(
        self,
        lora_weight_names: Tuple[Set[str]],
        lora_embeddings_weight_names: Tuple[Set[str]],
        base_model: torch.nn.Module,
        max_lora_dim: int,
        cur_max_extra_vocab_size: int,
    ):
        # lora_weight_names is a set of name pairs indicating each pair of lora modules to load
        #   e.g., {("qkv_proj", "q_proj"), ("qkv_proj", "kv_proj"), ("o_proj", "o_proj")}
        self.lora_weight_names: Tuple[Set[str]] = lora_weight_names
        self.lora_embeddings_weight_names: Tuple[Set[str]] = lora_embeddings_weight_names
        device = next(base_model.parameters()).device

        def update_buffer(
            buffer: Dict[str, List[torch.Tensor]],
            lora_weight_names: Set[str],
            get_lora_shape_fn: Callable[[str, torch.nn.Module, int], Tuple[int]],
        ):
            new_weight_names = lora_weight_names - buffer.keys()
            for module_name in new_weight_names:
                if "embed_tokens" in module_name:
                    lora_shape = get_lora_shape_fn(max_lora_dim)
                    buffer[module_name] = torch.empty(
                        lora_shape,
                        dtype=self.dtype,
                        device=device,
                    )
                else:
                    lora_shape = get_lora_shape_fn(module_name, base_model, max_lora_dim)
                    buffer[module_name] = [
                        torch.empty(
                            lora_shape,
                            dtype=self.dtype,
                            device=device,
                        )
                        for _ in range(self.num_layer)
                    ]

        if cur_max_extra_vocab_size != self.max_extra_vocab_size:
            self.max_extra_vocab_size = cur_max_extra_vocab_size
            self.new_embeddings_buffer["input_embeddings"] = torch.empty(
                (self.max_loras_per_batch, self.max_extra_vocab_size, self.embedding_dim),
                dtype=self.dtype,
                device=device,
            )
            self.embedding_A_buffer.clear()

        update_buffer(
            self.embedding_A_buffer,
            lora_embeddings_weight_names[0],
            self.get_embedding_lora_A_shape,
        )

        update_buffer(
            self.embedding_B_buffer,
            lora_embeddings_weight_names[1],
            self.get_embedding_lora_B_shape,
        )

        update_buffer(
            self.A_buffer,
            lora_weight_names[0],
            self.get_lora_A_shape,
        )

        update_buffer(
            self.B_buffer,
            lora_weight_names[1],
            self.get_lora_B_shape,
        )
        
    def prepare_lora_batch(
        self,
        cur_uids: Set[Optional[str]],
        lora_adapters: Dict[str, LoRAAdapter],
        lora_modules: Dict[int, Dict[str, BaseLayerWithLoRA]],
    ):
        def get_available_buffer_slot():
            for buffer_id in range(self.max_loras_per_batch):
                # Prioritize empty slots
                if self.buffer_id_to_uid[buffer_id] == "":
                    return buffer_id

            for buffer_id in range(self.max_loras_per_batch):
                # Evict unneeded lora
                if self.buffer_id_to_uid[buffer_id] not in cur_uids:
                    self.uid_to_buffer_id.pop(self.buffer_id_to_uid[buffer_id])
                    return buffer_id

            raise ValueError(
                "No available buffer slots found. Please ensure the number of active loras is less than max_loras_per_batch."
            )

        for uid in cur_uids:
            if uid not in self.uid_to_buffer_id:
                buffer_id = get_available_buffer_slot()
                lora_adapter = lora_adapters.get(uid, None)
                self.load_lora_weight_to_buffer(
                    uid, buffer_id, lora_adapter, lora_modules
                )
                self.uid_to_buffer_id[uid] = buffer_id
                self.buffer_id_to_uid[buffer_id] = uid

    def load_lora_weight_to_buffer(
        self,
        uid: str,
        buffer_id: int,
        lora_adapter: LoRAAdapter,
        lora_modules: Dict[int, Dict[str, BaseLayerWithLoRA]],
    ):
        def check_lora_weight_shape(buffer_view: torch.Tensor, weight: torch.Tensor):
            assert (
                buffer_view.shape == weight.shape
            ), f"LoRA buffer shape {buffer_view.shape} does not match weight shape {weight.shape}."

        if uid is None:
            for i in range(self.num_layer):
                for k in self.A_buffer.keys():
                    self.A_buffer[k][i][buffer_id] = 0
            return

        assert lora_adapter is not None
        lora_rank = lora_adapter.config.hf_config["r"]
        for layer_id in range(self.num_layer):
            layer_weights = lora_adapter.layers[layer_id].weights
            temp_A_buffer: Dict[str, torch.Tensor] = {}
            temp_B_buffer: Dict[str, torch.Tensor] = {}
            for name, weights in layer_weights.items():
                if "lora_A" in name:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_A
                    )
                    temp_A_buffer[lora_weight_name] = weights
                else:
                    lora_weight_name = get_weight_name(
                        name, self.lora_weight_names, LoRAType.LORA_B
                    )
                    temp_B_buffer[lora_weight_name] = weights

            if self.tp_size > 1:
                cur_layer_modules = lora_modules[layer_id]
                for module_name, module in cur_layer_modules.items():
                    if "qkv_proj" in module_name:
                        temp_A_buffer["qkv_proj"] = module.slice_lora_a_weights(
                            temp_A_buffer["qkv_proj"], self.tp_rank
                        )
                        temp_B_buffer["q_proj"], temp_B_buffer["kv_proj"] = (
                            module.slice_lora_b_weights(
                                [temp_B_buffer["q_proj"], temp_B_buffer["kv_proj"]],
                                self.tp_rank,
                            )
                        )
                    else:
                        weight_name = get_weight_name(
                            module_name, self.lora_weight_names, LoRAType.LORA_A
                        )
                        temp_A_buffer[weight_name] = module.slice_lora_a_weights(
                            temp_A_buffer[weight_name], self.tp_rank
                        )
                        temp_B_buffer[weight_name] = module.slice_lora_b_weights(
                            temp_B_buffer[weight_name], self.tp_rank
                        )

            for name, weights in temp_A_buffer.items():
                c = get_stacked_multiply(name)
                buffer_view = self.A_buffer[name][layer_id][buffer_id][
                    : lora_rank * c, :
                ]
                check_lora_weight_shape(buffer_view, weights)
                buffer_view.copy_(weights)

            for name, weights in temp_B_buffer.items():
                c = get_stacked_multiply(name)
                if c > 1:
                    for stacked_id in range(c):
                        buffer_view = self.B_buffer[name][layer_id][stacked_id][
                            buffer_id
                        ][:, :lora_rank]
                        check_lora_weight_shape(buffer_view, weights[stacked_id])
                        buffer_view.copy_(weights[stacked_id])
                else:
                    buffer_view = self.B_buffer[name][layer_id][0][buffer_id][
                        :, :lora_rank
                    ]
                    check_lora_weight_shape(buffer_view, weights)
                    buffer_view.copy_(weights)
        # Load embeddings weights to buffer
        org_vocab_size = self.base_hf_config.vocab_size
        extra_vocab_size = lora_adapter.extra_vocab_size
        if lora_adapter.new_embeddings:
            for name, weights in lora_adapter.new_embeddings.items():
                if "input_embeddings" in name:
                    buffer_view = self.new_embeddings_buffer["input_embeddings"][buffer_id, :extra_vocab_size]
                    check_lora_weight_shape(buffer_view, weights)
                    buffer_view.copy_(weights)
        
        if lora_adapter.weights:
            for name, weights in lora_adapter.weights.items():
                if "lora_embedding_A" in name:
                    lora_weight_name = get_weight_name(
                        name, self.lora_embeddings_weight_names, LoRAType.LORA_A
                    )
                    buffer_view = self.embedding_A_buffer[lora_weight_name][buffer_id, :lora_rank, :org_vocab_size+extra_vocab_size]
                    check_lora_weight_shape(buffer_view, weights)
                    buffer_view.copy_(weights)
                elif "lora_embedding_B" in name:
                    lora_weight_name = get_weight_name(
                        name, self.lora_embeddings_weight_names, LoRAType.LORA_B
                    )
                    lora_b_weights = weights
                    if self.tp_size > 1:
                        cur_module = self.lora_embeddings_modules[lora_weight_name]
                        for module_name, module in cur_module:
                            weight_name = get_weight_name(
                                module_name, self.lora_embeddings_weight_names, LoRAType.LORA_B
                            )
                            lora_b_weights = module.slice_lora_b_weights(lora_b_weights, self.tp_rank)
                    
                    buffer_view = self.embedding_B_buffer[lora_weight_name][buffer_id, :, :lora_rank]
                    check_lora_weight_shape(buffer_view, lora_b_weights)
                    buffer_view.copy_(lora_b_weights)

    def get_tensor(
        self, weight_name: str, layer_id: int, lora_type: LoRAType
    ) -> torch.Tensor:
        if lora_type == LoRAType.LORA_A:
            return self.A_buffer[weight_name][layer_id]

        return self.B_buffer[weight_name][layer_id]

    def get_embedding_tensor(
        self, weight_name: str, lora_type: Optional[LoRAType] = None
    ) -> torch.Tensor:
        if lora_type is None:
            return self.new_embeddings_buffer["input_embeddings"]
        if lora_type == LoRAType.LORA_A:
            return self.embedding_A_buffer[weight_name]
        return self.embedding_B_buffer[weight_name]

    def get_buffer_id(self, lora_uid: str):
        return self.uid_to_buffer_id[lora_uid]
