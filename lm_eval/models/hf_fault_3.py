import numpy as np
import torch
import random
import logging
import os
import csv
from typing import Optional, Union, List, Tuple, Any
from datetime import datetime

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf_fault_3")
class KVBitFlipHFLM(HFLM):
    """
    hf_fault_3: KV-cache bit-flip injector (BF16 only).

    Supported modes:
    - fault_mode="single": flip exactly one bit per generate request.
    - fault_mode="prob": sample the number of flips by inject_prob.

    bit_pos semantics:
    - bit_pos=None: randomly select bit positions from 0..15.
    - bit_pos=0..15: always flip the specified bf16 bit position.

    kv_target semantics:
    - kv_target="both": inject into both key cache and value cache.
    - kv_target="key": inject only into key cache.
    - kv_target="value": inject only into value cache.

    Injection timing:
    - The model forward is wrapped.
    - For each lm-eval generate request, injection happens when outputs.past_key_values
      first becomes available.
    - After a successful injection attempt for the request, later forward steps in the
      same request are not injected again.
    """

    def __init__(
        self,
        pretrained: str,
        inject_prob: float = 0.0,
        fault_mode: str = "single",
        bit_pos: Optional[Union[int, str]] = None,
        kv_target: str = "both",
        output_path: Optional[str] = None,
        seed: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        # Define early so __del__ is safe even if super().__init__ fails.
        self.original_forward = None
        self._log_buffer: List[List[Any]] = []
        self.total_flips = 0
        self.total_attempted_flips = 0
        self.total_available_bits = 0
        self.total_available_elements = 0

        super().__init__(pretrained=pretrained, **kwargs)

        # BF16 only. This is intentionally strict because the experimental target is bf16 LLMs.
        if getattr(self.model, "dtype", None) != torch.bfloat16:
            raise ValueError(
                f"[hf_fault_3] This injector only supports bf16. "
                f"Got model.dtype={getattr(self.model, 'dtype', None)}"
            )

        self.pretrained = pretrained
        self.inject_prob = float(inject_prob)
        self.fault_mode = str(fault_mode).lower().strip()
        self.bit_pos = self._normalize_bit_pos(bit_pos)
        self.kv_target = str(kv_target).lower().strip()
        self._validate_config()

        # Per-request state.
        self._request_idx = 0
        self._injected_this_request = False

        # Randomness.
        self._setup_random(seed)

        # CSV logging.
        self._init_logging(output_path)

        # Wrap forward: inject when outputs.past_key_values first becomes valid.
        self.original_forward = self.model.forward
        self.model.forward = self._instrumented_forward

        bit_mode = "random_bit_0_15" if self.bit_pos is None else f"fixed_bit_{self.bit_pos}"
        logging.info(
            f"[hf_fault_3] Initialized KV bit-flip injector "
            f"(bf16 only, fault_mode={self.fault_mode}, inject_prob={self.inject_prob}, "
            f"kv_target={self.kv_target}, bit_mode={bit_mode}, "
            f"seed_label={self.seed_label}). CSV: {self.log_file}"
        )

    # -------------------------------------------------------------------------
    # Request boundary
    # -------------------------------------------------------------------------
    def _model_generate(self, context, max_length, stop, **kwargs):
        """
        lm-eval generation requests go through this method.
        Reset the injection flag so each request has at most one injection event.
        """
        self._request_idx += 1
        self._injected_this_request = False
        return super()._model_generate(context, max_length, stop, **kwargs)

    # -------------------------------------------------------------------------
    # Forward wrapper
    # -------------------------------------------------------------------------
    def _instrumented_forward(self, *args, **kwargs):
        kwargs["use_cache"] = True
        kwargs["output_attentions"] = False

        outputs = self.original_forward(*args, **kwargs)

        if (
            (not self._injected_this_request)
            and hasattr(outputs, "past_key_values")
            and outputs.past_key_values is not None
        ):
            try:
                # Mark this request as handled even if prob mode samples zero flips.
                # This preserves the semantics: one injection opportunity per request.
                attempted = self._inject_kv_cache_fault(outputs.past_key_values)
                if attempted:
                    self._injected_this_request = True
            except Exception as e:
                logging.warning(f"[hf_fault_3] Injection failed in forward: {e}")

        return outputs

    # -------------------------------------------------------------------------
    # Config and randomness
    # -------------------------------------------------------------------------
    def _normalize_bit_pos(self, bit_pos: Optional[Union[int, str]]) -> Optional[int]:
        """
        bit_pos=None means random bit position 0..15.
        Also accepts strings like "None", "none", "null", or empty string as None.
        """
        if bit_pos is None:
            return None
        if isinstance(bit_pos, str):
            s = bit_pos.strip().lower()
            if s in ("", "none", "null"):
                return None
            return int(s)
        return int(bit_pos)

    def _validate_config(self) -> None:
        if self.fault_mode not in ("single", "prob"):
            raise ValueError(
                f"[hf_fault_3] Invalid fault_mode={self.fault_mode!r}. "
                f"Expected 'single' or 'prob'."
            )

        if not 0 <= self.inject_prob <= 1:
            raise ValueError(
                f"[hf_fault_3] Invalid inject_prob={self.inject_prob}. Must be in [0, 1]."
            )

        if self.bit_pos is not None and not (0 <= self.bit_pos < 16):
            raise ValueError(
                f"[hf_fault_3] Invalid bit_pos={self.bit_pos}. "
                f"Expected None for random bit or 0..15 for fixed bf16 bit."
            )

        if self.kv_target not in ("both", "key", "value"):
            raise ValueError(
                f"[hf_fault_3] Invalid kv_target={self.kv_target!r}. "
                f"Expected 'both', 'key', or 'value'."
            )

    def _generate_unique_seed(self) -> int:
        now = datetime.now()
        timestamp_seed = int(now.timestamp() * 1e6)
        pid_seed = os.getpid()
        system_random = random.SystemRandom()
        random_seed = system_random.randint(0, 2**32 - 1)
        return (timestamp_seed ^ pid_seed ^ random_seed) & 0xFFFFFFFF

    def _setup_random(self, seed: Optional[Union[int, str]]) -> None:
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
                        f"[hf_fault_3] Unsupported seed string: {seed!r}, "
                        f"expected 'random' or integer-like."
                    )
        else:
            self.seed = int(seed)

        self.random_state = random.Random(self.seed)
        self.np_random_state = np.random.RandomState(self.seed % (2**32))
        self.seed_label = str(self.seed)

    # -------------------------------------------------------------------------
    # CSV logging
    # -------------------------------------------------------------------------
    def _init_logging(self, output_path: Optional[str] = None) -> None:
        if output_path:
            log_dir = os.path.dirname(output_path)
            if not log_dir:
                log_dir = "."
        else:
            log_dir = "./results/3"

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().isoformat().replace(":", "-")
        self.csv_filename = f"results_{timestamp}.csv"
        self.log_file = os.path.join(log_dir, self.csv_filename)

        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "seed_label",
                    "request_idx",
                    "fault_mode",
                    "inject_prob",
                    "kv_target",
                    "layer_idx",
                    "head_idx",
                    "seq_pos",
                    "kv_type",
                    "elem_idx",
                    "bit_pos",
                    "original_value",
                    "modified_value",
                ]
            )

        logging.info(f"[hf_fault_3] Fault injection log will be saved to: {self.log_file}")

    def _log_fault(
        self,
        layer_idx: int,
        head_idx: int,
        seq_pos: int,
        kv_type: str,
        elem_idx: int,
        bit_pos: int,
        original_val: float,
        modified_val: float,
    ) -> None:
        if original_val == modified_val:
            return

        row = [
            self.seed_label,
            self._request_idx,
            self.fault_mode,
            self.inject_prob,
            self.kv_target,
            layer_idx,
            head_idx,
            seq_pos,
            kv_type,
            elem_idx,
            bit_pos,
            float(original_val),
            float(modified_val),
        ]

        try:
            with open(self.log_file, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            logging.error(f"[hf_fault_3] Failed to write CSV row: {e}")

    # -------------------------------------------------------------------------
    # BF16 bit flip
    # -------------------------------------------------------------------------
    def _bit_flip_bf16(self, value: float, bit_pos: int) -> float:
        """
        Flip bit_pos in the bf16 representation.
        float32 high 16 bits are equivalent to bf16 bits.
        """
        bit_pos = int(bit_pos)
        if not (0 <= bit_pos < 16):
            raise ValueError(f"[hf_fault_3] bf16 bit_pos must be 0..15, got {bit_pos}")

        f32 = np.array([value], dtype=np.float32)
        u32 = f32.view(np.uint32)
        u32_val = int(u32.reshape(-1)[0].item())

        bf16_u16 = (u32_val >> 16) & 0xFFFF
        bf16_u16 ^= (1 << bit_pos)

        new_u32_val = (bf16_u16 & 0xFFFF) << 16
        new_u32 = np.array([new_u32_val], dtype=np.uint32)
        new_f32 = new_u32.view(np.float32)
        return float(new_f32.reshape(-1)[0].item())

    # -------------------------------------------------------------------------
    # KV cache collection and mapping
    # -------------------------------------------------------------------------
    def _iter_past_key_values(self, past_key_values):
        """
        Return an iterable layer cache.

        In the tested environment, DynamicCache is directly iterable and yields:
            (key_tensor, value_tensor)

        This fallback keeps compatibility with cache objects that expose
        to_legacy_cache().
        """
        try:
            return list(past_key_values)
        except Exception:
            if hasattr(past_key_values, "to_legacy_cache"):
                return list(past_key_values.to_legacy_cache())
            raise

    def _collect_kv_views(self, past_key_values) -> List[Tuple[int, str, torch.Tensor]]:
        """
        Collect valid KV tensors.

        Expected format:
            past_key_values[layer_idx] = (key_tensor, value_tensor)

        Expected tensor shape:
            [batch, num_key_value_heads, seq_len, head_dim]

        Important:
        - kv_target filtering happens here.
        - Therefore total_bits and total_elements are computed over the selected
          injection space, not always over the full K+V cache.
        """
        views: List[Tuple[int, str, torch.Tensor]] = []

        layers = self._iter_past_key_values(past_key_values)

        for layer_idx, layer in enumerate(layers):
            try:
                layer_items = list(layer)
            except Exception as e:
                logging.warning(
                    f"[hf_fault_3] Cannot iterate layer cache, skip. "
                    f"layer={layer_idx}, type={type(layer)}, err={e}"
                )
                continue

            if len(layer_items) < 2:
                logging.warning(
                    f"[hf_fault_3] Layer cache has fewer than 2 items, skip. "
                    f"layer={layer_idx}, len={len(layer_items)}"
                )
                continue

            # Only the first two entries are treated as key/value.
            # Verified format for Qwen3-8B and Llama-3.1-8B-Instruct:
            # layer = (key_tensor, value_tensor)
            for kv_type_idx, kv_tensor in enumerate(layer_items[:2]):
                kv_type = "key" if kv_type_idx == 0 else "value"

                # The key fix:
                # Filter K/V before appending to views. All later space calculation
                # and sampling are based on this filtered views list.
                if self.kv_target == "key" and kv_type != "key":
                    continue
                if self.kv_target == "value" and kv_type != "value":
                    continue

                if kv_tensor is None:
                    continue
                if not isinstance(kv_tensor, torch.Tensor):
                    logging.warning(
                        f"[hf_fault_3] KV item is not Tensor, skip. "
                        f"layer={layer_idx}, kv_type={kv_type}, type={type(kv_tensor)}"
                    )
                    continue
                if kv_tensor.numel() == 0:
                    continue
                if kv_tensor.dtype != torch.bfloat16:
                    raise ValueError(
                        f"[hf_fault_3] KV tensor must be bf16, got {kv_tensor.dtype} "
                        f"at layer={layer_idx}, kv_type={kv_type}."
                    )

                views.append((layer_idx, kv_type, kv_tensor))

        return views

    def _calculate_kv_space(self, views: List[Tuple[int, str, torch.Tensor]]) -> Tuple[int, int]:
        """
        Calculate the selected injection space.

        If kv_target="key", views only contains key tensors.
        If kv_target="value", views only contains value tensors.
        If kv_target="both", views contains both key and value tensors.
        """
        total_elements = int(sum(kv_tensor.numel() for _, _, kv_tensor in views))
        total_bits = total_elements * 16
        return total_bits, total_elements

    def _sample_flip_count(self, total_bits: int, total_elements: int) -> int:
        if self.fault_mode == "single":
            return 1

        # prob mode
        if self.bit_pos is None:
            # Each bf16 bit in the selected injection space is independently sampled.
            return int(self.np_random_state.binomial(total_bits, self.inject_prob))
        else:
            # Each bf16 element in the selected injection space contributes only
            # the fixed bit_pos.
            return int(self.np_random_state.binomial(total_elements, self.inject_prob))

    def _generate_flip_mapping(
        self,
        views: List[Tuple[int, str, torch.Tensor]],
        flip_count: int,
        total_bits: int,
        total_elements: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Return a list of (view_idx, local_element_index, bit_pos).

        If bit_pos=None:
            sample globally from all bits in the selected KV space.

        If bit_pos is fixed:
            sample globally from all elements in the selected KV space,
            and always flip that fixed bit.
        """
        if flip_count <= 0:
            return []

        if self.bit_pos is None:
            counts = np.array(
                [kv_tensor.numel() * 16 for _, _, kv_tensor in views],
                dtype=np.int64,
            )
            positions = self.np_random_state.randint(0, total_bits, size=flip_count)
            cum = np.cumsum(counts)
            view_indices = np.searchsorted(cum, positions, side="right")
            prev_cum = np.concatenate(([0], cum[:-1]))
            local_bit_offsets = positions - prev_cum[view_indices]

            mapping: List[Tuple[int, int, int]] = []
            for vi, offset in zip(view_indices, local_bit_offsets):
                elem_idx = int(offset // 16)
                bit_pos = int(offset % 16)
                mapping.append((int(vi), elem_idx, bit_pos))
            return mapping

        counts = np.array(
            [kv_tensor.numel() for _, _, kv_tensor in views],
            dtype=np.int64,
        )
        elem_positions = self.np_random_state.randint(0, total_elements, size=flip_count)
        cum = np.cumsum(counts)
        view_indices = np.searchsorted(cum, elem_positions, side="right")
        prev_cum = np.concatenate(([0], cum[:-1]))
        local_elem_indices = elem_positions - prev_cum[view_indices]

        return [
            (int(vi), int(elem_idx), int(self.bit_pos))
            for vi, elem_idx in zip(view_indices, local_elem_indices)
        ]

    # -------------------------------------------------------------------------
    # KV cache injection
    # -------------------------------------------------------------------------
    def _inject_kv_cache_fault(self, past_key_values) -> bool:
        """
        Returns True once an injection opportunity has been handled.

        In prob mode, sampling zero flips is still a handled opportunity.
        """
        if past_key_values is None:
            return False

        views = self._collect_kv_views(past_key_values)
        if not views:
            logging.warning(
                f"[hf_fault_3] No valid KV tensors found "
                f"(request_idx={self._request_idx}, kv_target={self.kv_target})"
            )
            return False

        total_bits, total_elements = self._calculate_kv_space(views)
        self.total_available_bits += total_bits
        self.total_available_elements += total_elements

        flip_count = self._sample_flip_count(total_bits, total_elements)
        self.total_attempted_flips += flip_count

        logging.info(
            f"[hf_fault_3] KV injection opportunity for request_idx={self._request_idx}: "
            f"fault_mode={self.fault_mode}, kv_target={self.kv_target}, "
            f"selected_views={len(views)}, total_elements={total_elements:,}, "
            f"total_bits={total_bits:,}, sampled_flips={flip_count}"
        )

        if flip_count == 0:
            return True

        mapping = self._generate_flip_mapping(views, flip_count, total_bits, total_elements)
        actual_flips = 0

        for view_idx, local_elem_idx, actual_bit_pos in mapping:
            layer_idx, kv_type, kv_tensor = views[view_idx]

            try:
                bsz, num_kv_heads, seq_len, head_dim = kv_tensor.shape
            except Exception as e:
                logging.warning(
                    f"[hf_fault_3] Unexpected KV tensor shape, skip. "
                    f"shape={getattr(kv_tensor, 'shape', None)}, err={e}"
                )
                continue

            if bsz < 1 or num_kv_heads < 1 or seq_len < 1 or head_dim < 1:
                logging.warning(
                    f"[hf_fault_3] Invalid KV tensor shape, skip. "
                    f"shape={kv_tensor.shape}"
                )
                continue

            # local_elem_idx is within the flattened tensor including batch dimension.
            # lm-eval generation normally uses batch size 1, but this keeps mapping exact.
            b_idx = local_elem_idx // (num_kv_heads * seq_len * head_dim)
            rem = local_elem_idx % (num_kv_heads * seq_len * head_dim)
            head_idx = rem // (seq_len * head_dim)
            rem = rem % (seq_len * head_dim)
            seq_pos = rem // head_dim
            elem_idx = rem % head_dim

            try:
                original_val = kv_tensor[b_idx, head_idx, seq_pos, elem_idx].item()
                modified_val = self._bit_flip_bf16(original_val, actual_bit_pos)
                kv_tensor[b_idx, head_idx, seq_pos, elem_idx] = modified_val
            except Exception as e:
                logging.warning(
                    f"[hf_fault_3] Failed to flip KV value: {e} "
                    f"(request_idx={self._request_idx}, layer={layer_idx}, "
                    f"kv_target={self.kv_target}, kv_type={kv_type}, "
                    f"flat_elem={local_elem_idx}, bit={actual_bit_pos})"
                )
                continue

            if original_val != modified_val:
                actual_flips += 1
                self.total_flips += 1
                self._log_fault(
                    layer_idx=layer_idx,
                    head_idx=int(head_idx),
                    seq_pos=int(seq_pos),
                    kv_type=kv_type,
                    elem_idx=int(elem_idx),
                    bit_pos=int(actual_bit_pos),
                    original_val=original_val,
                    modified_val=modified_val,
                )

        logging.info(
            f"[hf_fault_3] Finished KV injection for request_idx={self._request_idx}: "
            f"kv_target={self.kv_target}, attempted={flip_count}, actual={actual_flips}"
        )
        return True

    # -------------------------------------------------------------------------
    # lm-eval interface compatibility
    # -------------------------------------------------------------------------
    def _model_call(self, inputs, **kwargs):
        """
        Classification/loglikelihood tasks may go through _model_call.
        The target of hf_fault_3 is KV-cache faults during generation, so no extra
        request-boundary handling is added here.
        """
        call_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ["use_cache", "output_attentions"]
        }
        return super()._model_call(inputs, **call_kwargs)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def __del__(self):
        try:
            if hasattr(self, "total_flips"):
                logging.info(f"[hf_fault_3] Total actual flips: {self.total_flips}")
            if hasattr(self, "total_attempted_flips"):
                logging.info(f"[hf_fault_3] Total attempted flips: {self.total_attempted_flips}")
            if hasattr(self, "total_available_elements"):
                logging.info(
                    f"[hf_fault_3] Total available elements in selected KV space: "
                    f"{self.total_available_elements}"
                )
            if hasattr(self, "total_available_bits"):
                logging.info(
                    f"[hf_fault_3] Total available bits in selected KV space: "
                    f"{self.total_available_bits}"
                )
            if hasattr(self, "log_file"):
                logging.info(f"[hf_fault_3] CSV saved to: {self.log_file}")
        except Exception:
            pass

        if getattr(self, "original_forward", None) is not None and hasattr(self, "model"):
            try:
                self.model.forward = self.original_forward
            except Exception:
                pass