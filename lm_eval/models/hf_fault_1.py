import torch
import random
import logging
import numpy as np
import os
import csv
from datetime import datetime
from typing import List, Optional, Union

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf_fault_1")
class BitFlipHFLM(HFLM):
    """
    hf_fault_1: BF16 activation bit-flip injector.

    fault_mode:
    - "prob":   probabilistic bit-flip injection.
    - "single": inject exactly one bit per hooked activation tensor.

    bit_pos:
    - None: randomly choose a bit from 0..15.
    - 0..15: fixed bf16 bit position.

    layers:
    - "all": all transformer layers.
    - "3": layer 3.
    - "3:5": layers 3,4,5.
    - "1,3,5": selected layers.
    """

    def __init__(
        self,
        pretrained: str,
        inject_prob: float = 0,
        fault_mode: str = "prob",
        layers: Union[str, int] = "all",
        bit_pos: Optional[int] = None,
        output_path: Optional[str] = None,
        seed: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        self.log_batch: List[List] = []
        self.hook_handles: List = []

        super().__init__(pretrained=pretrained, **kwargs)

        self._validate_config(inject_prob, fault_mode, bit_pos)

        self.inject_prob = float(inject_prob)
        self.fault_mode = str(fault_mode).lower()
        self.bit_pos = None if bit_pos is None else int(bit_pos)
        self.target_layers = self._parse_layers(layers)

        self.fault_counter = 0
        self.total_available_flips = 0
        self.log_batch_size = 1000

        if output_path:
            self.output_dir = os.path.dirname(output_path)
        else:
            self.output_dir = "./results"

        self.log_file = self._create_log_file()
        self._init_logging()

        self._setup_random(seed)
        self._register_layer_hooks()

        bit_mode = "random_bit_0_15" if self.bit_pos is None else f"fixed_bit_{self.bit_pos}"
        layer_mode = "all" if not self.target_layers else str(self.target_layers)

        logging.info(
            f"Initialized bf16 activation bit-flip injector "
            f"(fault_mode={self.fault_mode}, prob={self.inject_prob}, "
            f"bit_mode={bit_mode}, layers={layer_mode}, seed_label={self.seed_label})"
        )
        logging.info(f"Output directory: {self.output_dir}")

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    def _validate_config(
        self,
        inject_prob: float,
        fault_mode: str,
        bit_pos: Optional[int],
    ):
        if not 0 <= float(inject_prob) <= 1:
            raise ValueError(f"Invalid inject_prob: {inject_prob}. Must be in [0,1].")

        fault_mode = str(fault_mode).lower()
        if fault_mode not in ("prob", "single"):
            raise ValueError(
                f"Invalid fault_mode: {fault_mode}. Expected 'prob' or 'single'."
            )

        if bit_pos is not None:
            bit_pos = int(bit_pos)
            if not 0 <= bit_pos < 16:
                raise ValueError(
                    f"Invalid bit_pos: {bit_pos}. For bf16, expected None or 0..15."
                )

    # -------------------------------------------------------------------------
    # Randomness
    # -------------------------------------------------------------------------
    def _generate_unique_seed(self) -> int:
        now = datetime.now()
        timestamp_seed = int(now.timestamp() * 1e6)
        pid_seed = os.getpid()
        system_random = random.SystemRandom()
        random_seed = system_random.randint(0, 2**32 - 1)
        return (timestamp_seed ^ pid_seed ^ random_seed) & 0xFFFFFFFF

    def _setup_random(self, seed: Optional[Union[int, str]]) -> None:
        """
        seed=None:
            Follow lm-eval / framework global RNG.
        seed='random':
            Generate a unique seed for this run.
        seed=int or str-int:
            Use an independent reproducible RNG.
        """
        self.seed: Optional[int] = None

        if seed is None:
            self.random_state = random
            self.np_random_state = np.random
            self.seed_label = "framework_default"
            return

        if isinstance(seed, str):
            if seed.lower() == "random":
                self.seed = self._generate_unique_seed()
            else:
                try:
                    self.seed = int(seed)
                except Exception:
                    raise ValueError(
                        f"Unsupported seed string: {seed!r}, expected 'random' or integer-like."
                    )
        else:
            self.seed = int(seed)

        self.random_state = random.Random(self.seed)
        self.np_random_state = np.random.RandomState(self.seed % (2**32))
        self.seed_label = str(self.seed)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    def _init_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Fault injection log: {self.log_file}")

    def _create_log_file(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="microseconds").replace(":", "-")
        filename = f"results_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seed",
                    "fault_mode",
                    "layer_idx",
                    "element_index",
                    "bit_pos",
                    "original_float",
                    "modified_float",
                ]
            )

        return filepath

    def _log_fault(
        self,
        layer_idx: int,
        element_index: int,
        bit_pos: int,
        original_val: float,
        modified_val: float,
    ):
        self.log_batch.append(
            [
                self.seed_label,
                self.fault_mode,
                layer_idx,
                element_index,
                bit_pos,
                original_val,
                modified_val,
            ]
        )

        if len(self.log_batch) >= self.log_batch_size:
            self._flush_log_batch()

    def _flush_log_batch(self):
        if not getattr(self, "log_batch", None):
            return

        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.log_batch)
            self.log_batch = []
        except Exception as e:
            logging.error(f"Failed to write fault log: {e}")

    # -------------------------------------------------------------------------
    # Main bit-flip logic
    # -------------------------------------------------------------------------
    def _bit_flip_tensor(self, tensor: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Inject bit flips into one activation tensor.

        Only bf16 tensors are modified.
        """
        if tensor.dtype != torch.bfloat16:
            return tensor

        total_elements = tensor.numel()
        if total_elements == 0:
            return tensor

        bits_per_element = 16
        self.total_available_flips += total_elements * bits_per_element

        if self.fault_mode == "single":
            flip_count = 1
        else:
            if self.bit_pos is None:
                flip_count = int(
                    self.np_random_state.binomial(
                        total_elements * bits_per_element,
                        self.inject_prob,
                    )
                )
            else:
                flip_count = int(
                    self.np_random_state.binomial(
                        total_elements,
                        self.inject_prob,
                    )
                )

        if flip_count == 0:
            return tensor

        modified = tensor.clone()
        flat_tensor = modified.view(-1)

        if self.bit_pos is None:
            positions = self.np_random_state.randint(
                0,
                total_elements * bits_per_element,
                size=flip_count,
            )
            element_indices = positions // bits_per_element
            bit_indices = positions % bits_per_element
        else:
            element_indices = self.np_random_state.randint(
                0,
                total_elements,
                size=flip_count,
            )
            bit_indices = np.full(
                shape=(flip_count,),
                fill_value=self.bit_pos,
                dtype=np.int64,
            )

        unique_bit_positions = np.unique(bit_indices)

        for bpos in unique_bit_positions:
            mask = bit_indices == bpos
            indices_for_bit = element_indices[mask]
            if len(indices_for_bit) > 0:
                self._batch_process_bfloat16_high16(
                    flat_tensor,
                    indices_for_bit.astype(np.int64),
                    int(bpos),
                    layer_idx,
                )

        return modified

    def _batch_process_bfloat16_high16(
        self,
        tensor: torch.Tensor,
        indices: np.ndarray,
        bit_pos: int,
        layer_idx: int,
    ):
        """
        BF16 bit flip by manipulating the high 16 bits of float32 representation.

        bf16 bits correspond to float32 high 16 bits:
        - bf16 bit 15: sign
        - bf16 bits 14..7: exponent
        - bf16 bits 6..0: mantissa
        """
        if len(indices) == 0:
            return

        indices_tensor = torch.tensor(indices, device=tensor.device, dtype=torch.long)

        selected_values = tensor[indices_tensor]
        original_vals = selected_values.detach().clone().cpu().to(torch.float32)

        arr32 = original_vals.numpy().astype(np.float32)
        flat32 = arr32.reshape(-1)
        int_view32 = flat32.view(np.uint32)

        int_view32 ^= np.uint32(1 << (bit_pos + 16))

        modified_vals_f32 = torch.from_numpy(arr32.astype(np.float32))
        modified_vals_bf16 = modified_vals_f32.to(device=tensor.device).to(torch.bfloat16)

        tensor[indices_tensor] = modified_vals_bf16

        modified_vals_cpu = modified_vals_bf16.detach().cpu().to(torch.float32)

        for i, idx in enumerate(indices):
            original_val = float(original_vals[i].item())
            modified_val = float(modified_vals_cpu[i].item())

            self._log_fault(
                layer_idx=layer_idx,
                element_index=int(idx),
                bit_pos=bit_pos,
                original_val=original_val,
                modified_val=modified_val,
            )
            self.fault_counter += 1

    # -------------------------------------------------------------------------
    # Hooks and layer parsing
    # -------------------------------------------------------------------------
    def _layer_hook(self, module, inputs, outputs, layer_idx: int):
        if self.target_layers and layer_idx not in self.target_layers:
            return outputs

        if isinstance(outputs, tuple):
            if not isinstance(outputs[0], torch.Tensor):
                return outputs
            modified_output = self._bit_flip_tensor(outputs[0], layer_idx)
            return (modified_output,) + outputs[1:]

        if isinstance(outputs, torch.Tensor):
            return self._bit_flip_tensor(outputs, layer_idx)

        return outputs

    def _register_layer_hooks(self):
        layers = None
        model_attr_names = ["layers", "h", "encoder", "decoder", "transformer", "model"]

        for attr_name in model_attr_names:
            layers = getattr(self.model, attr_name, None)
            if layers is not None:
                break

            if hasattr(self.model, "model"):
                layers = getattr(self.model.model, attr_name, None)
                if layers is not None:
                    break

        if not layers:
            layers = []
            for name, module in self.model.named_modules():
                if any(keyword in name.lower() for keyword in ["layer", "block", "transformer"]):
                    if hasattr(module, "__len__") and len(module) > 0:
                        layers = module
                        break

            if not layers:
                raise ValueError(
                    "Unsupported model architecture - could not find transformer layers."
                )

        if self.target_layers:
            target_set = set(self.target_layers)

            for idx, layer in enumerate(layers):
                if idx in target_set:
                    handle = layer.register_forward_hook(self._make_hook(idx))
                    self.hook_handles.append(handle)
        else:
            for idx, layer in enumerate(layers):
                handle = layer.register_forward_hook(self._make_hook(idx))
                self.hook_handles.append(handle)

        logging.info(f"Registered activation hooks: {len(self.hook_handles)}")

    def _make_hook(self, layer_index: int):
        def hook_wrapper(module, inputs, outputs):
            return self._layer_hook(module, inputs, outputs, layer_index)

        return hook_wrapper

    def _parse_layers(self, layers: Union[str, int]) -> List[int]:
        if isinstance(layers, int):
            return [layers]

        layer_str = str(layers)
        if not layer_str or layer_str.lower() == "all":
            return []

        try:
            if ":" in layer_str:
                start, end = map(int, layer_str.split(":"))
                return list(range(start, end + 1))
            return [int(x.strip()) for x in layer_str.split(",") if x.strip()]
        except ValueError:
            raise ValueError(
                f"Invalid layers format: {layers}. "
                f"Expected integer, 'all', '3:5', or '1,3,5'."
            )

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def __del__(self):
        for handle in getattr(self, "hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass

        try:
            if hasattr(self, "log_batch") and hasattr(self, "log_file"):
                self._flush_log_batch()
        except Exception:
            pass

        try:
            if hasattr(self, "fault_counter") and hasattr(self, "total_available_flips"):
                print(
                    f"\nact_flips={self.fault_counter}"
                    f"\navl_flips={self.total_available_flips}"
                )
        except Exception:
            pass