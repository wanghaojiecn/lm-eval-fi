import os
import csv
import json
import random
import logging
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Union, Any

import numpy as np
import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf_fault_2")
class ParamBitFlipHFLM(HFLM):
    def __init__(
        self,
        pretrained: str,
        inject_prob: float = 0,
        modules: Optional[str] = None,
        bit_pos: Optional[int] = None,
        element_index: Optional[Union[int, str]] = None,
        fault_mode: str = "prob",   # "prob" / "single"
        output_path: Optional[str] = None,
        seed: Optional[Union[int, str]] = None,
        case_output_path: Optional[str] = None,
        record_doc: bool = False,
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)

        self.inject_prob = float(inject_prob)
        self.bit_pos = None if bit_pos is None else int(bit_pos)
        self.fault_mode = self._normalize_fault_mode(fault_mode)
        self.element_index = self._parse_optional_int(element_index, "element_index")
        self.modules = None if modules is None else [
            m.strip() for m in str(modules).split(",") if m.strip()
        ]
        if self.modules == []:
            self.modules = None

        self.record_doc = self._str_to_bool(record_doc)

        self._validate_config(self.inject_prob, self.bit_pos, modules, self.fault_mode, self.element_index,)

        # Randomness control
        self._setup_random(seed)

        # Runtime state
        self._injected = False
        self.fault_counter = 0
        self.total_available_bits = 0
        self.actual_flips = 0
        self._log_buffer = []

        # Case-level output logging state
        self._case_log_buffer: List[List[Any]] = []
        self._case_counter = 0

        # CSV output directory
        if output_path:
            csv_dir = os.path.dirname(output_path)
            os.makedirs(csv_dir, exist_ok=True)
            logging.info(f"Using output directory from command line: {csv_dir}")
        else:
            csv_dir = "./results/2"
            os.makedirs(csv_dir, exist_ok=True)
            logging.warning(f"No output_path provided, using default directory: {csv_dir}")

        # Original fault-location CSV
        self.log_file = self._create_log_file(csv_dir)

        # New case-level output CSV
        if case_output_path:
            case_dir = os.path.dirname(case_output_path)
            if case_dir:
                os.makedirs(case_dir, exist_ok=True)
            self.case_log_file = case_output_path
            self._init_case_log_file(self.case_log_file)
        else:
            self.case_log_file = self._create_case_log_file(csv_dir)

        self._init_logging()

        # Register one-time pre-forward hook.
        # The hook is removed once it fires, so parameter faults are injected once
        # before the first forward pass and then persist for the whole evaluation.
        self._hook_handles = []
        try:
            handle = self.model.register_forward_pre_hook(self._one_time_inject_hook)
            self._hook_handles.append(handle)
        except Exception as e:
            logging.exception("Failed to register forward pre hook: %s", e)

        target = "all bf16 parameters" if self.modules is None else f"modules={self.modules}"
        logging.info(
            "Initialized bf16 param bit-flip injector "
            f"(mode={self.fault_mode}, prob={self.inject_prob}, bit_pos={self.bit_pos}, "
            f"element_index={self.element_index}, "
            f"target={target}, seed_label={self.seed_label}, output_dir={csv_dir}, "
            f"fault_log_file={self.log_file}, case_log_file={self.case_log_file})"
        )

    # -------------------------------------------------------------------------
    # Config and randomness
    # -------------------------------------------------------------------------
    def _normalize_fault_mode(self, fault_mode: str) -> str:
        mode = str(fault_mode).strip().lower()
        single_aliases = {"single", "single_bit", "one", "one_bit", "1bit"}
        prob_aliases = {"prob", "probability", "binomial", "rate"}
        if mode in single_aliases:
            return "single"
        if mode in prob_aliases:
            return "prob"
        raise ValueError(
            f"Unsupported fault_mode: {fault_mode!r}. "
            "Use fault_mode=prob or fault_mode=single."
        )

    def _validate_config(
        self,
        inject_prob: float,
        bit_pos: Optional[int],
        modules: Optional[str],
        fault_mode: str,
        element_index: Optional[int],
    ):
        if not 0 <= float(inject_prob) <= 1:
            raise ValueError(f"Invalid inject_prob: {inject_prob}. Must be in [0,1]")

        # We only support bf16 here, so valid bit positions are 0..15.
        if bit_pos is not None and not 0 <= int(bit_pos) < 16:
            raise ValueError(f"Invalid bit_pos: {bit_pos}. For bf16, bit_pos must be in [0,15]")

        if modules is not None and not isinstance(modules, str):
            raise ValueError("modules should be a comma-separated string or None")

        if fault_mode not in {"prob", "single"}:
            raise ValueError("fault_mode must be 'prob' or 'single'")

        if element_index is not None:
            if fault_mode != "single":
                logging.warning(
                    "element_index is provided but fault_mode is not 'single'; "
                    "element_index will be ignored."
                )
            if int(element_index) < 0:
                raise ValueError(f"Invalid element_index: {element_index}. Must be >= 0")
            # if bit_pos is None:
            #     raise ValueError(
            #         "element_index requires bit_pos to be specified, because element_index "
            #         "only fixes the tensor element, not the bit position."
            #     )

    def _generate_unique_seed(self) -> int:
        now = datetime.now()
        timestamp_seed = int(now.timestamp() * 1e6)
        pid_seed = os.getpid()
        combined = hash((timestamp_seed, pid_seed, id(self)))
        return combined & 0xFFFFFFFF

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
                        f"Unsupported seed string: {seed!r}, expected 'random' or integer-like"
                    )
        else:
            self.seed = int(seed)

        self.random_state = random.Random(self.seed)
        self.np_random_state = np.random.RandomState(self.seed % (2**32))
        self.seed_label = str(self.seed)

    def _str_to_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _parse_optional_int(self, value: Any, name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value.lower() in {"", "none", "null"}:
                return None
        try:
            return int(value)
        except Exception:
            raise ValueError(f"Invalid {name}: {value!r}. Expected an integer or None.")

    # -------------------------------------------------------------------------
    # Original fault-location CSV logging
    # -------------------------------------------------------------------------
    def _create_log_file(self, directory: str) -> str:
        try:
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().isoformat().replace(":", "-")
            filename = f"results_{timestamp}.csv"
            filepath = os.path.join(directory, filename)

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "seed",
                    "module_category",
                    "param_name",
                    "param_shape",
                    "element_index",
                    "bit_pos",
                    "original_float",
                    "modified_float",
                ])

            logging.info(f"Created fault injection log: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"Failed to create log file in {directory}: {e}")
            temp_file = tempfile.mktemp(suffix=".csv", dir=directory)
            logging.warning(f"Using temporary file: {temp_file}")
            return temp_file

    def _log_fault(
        self,
        param_name: str,
        param_shape: torch.Size,
        element_idx: int,
        bit_pos: int,
        original_val: float,
        modified_val: float,
        module_category: str,
    ):
        try:
            if original_val != modified_val:
                self._log_buffer.append([
                    self.seed_label,
                    module_category,
                    param_name,
                    str(tuple(param_shape)),
                    int(element_idx),
                    int(bit_pos),
                    float(original_val),
                    float(modified_val),
                ])
                self.actual_flips += 1

                if len(self._log_buffer) >= 100:
                    self._flush_log_buffer()

        except Exception as e:
            logging.exception("Failed to log fault for %s: %s", param_name, e)

    def _flush_log_buffer(self):
        if self._log_buffer:
            try:
                with open(self.log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(self._log_buffer)
                self.fault_counter += len(self._log_buffer)
                self._log_buffer.clear()
                logging.debug("Flushed log entries to fault-location file")
            except Exception as e:
                logging.exception("Failed to flush log buffer: %s", e)

    def _init_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # -------------------------------------------------------------------------
    # New case-level output CSV logging
    # -------------------------------------------------------------------------
    def _case_header(self) -> List[str]:
        """
        Keep the fault-run case CSV columns exactly aligned with hf_record.py.
        """
        return [
            "task_name",
            "doc_id",
            "request_idx",
            "prompt",
            "continuation",
            "loglikelihood",
            "predicted_continuation",
            "generated_text",
            "generation_kwargs",
            "standard_answer_full",
            "standard_answer_final",
        ]

    def _create_case_log_file(self, directory: str) -> str:
        try:
            os.makedirs(directory, exist_ok=True)
            timestamp = datetime.now().isoformat().replace(":", "-")
            filename = f"case_outputs_fault_{timestamp}.csv"
            filepath = os.path.join(directory, filename)

            self._init_case_log_file(filepath)

            logging.info(f"Created case output log: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"Failed to create case output log file in {directory}: {e}")
            temp_file = tempfile.mktemp(suffix=".csv", dir=directory)
            self._init_case_log_file(temp_file)
            logging.warning(f"Using temporary case output file: {temp_file}")
            return temp_file

    def _init_case_log_file(self, filepath: str):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._case_header())

    def _json_dumps_safe(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, default=str)
        except Exception:
            return str(obj)

    def _get_request_args(self, request: Any) -> Tuple[Any, ...]:
        """
        Compatible with lm-eval Instance objects and plain tuple/list requests.
        Different lm-eval versions may use .args or .arguments.
        """
        if hasattr(request, "args"):
            return tuple(request.args)
        if hasattr(request, "arguments"):
            return tuple(request.arguments)
        if isinstance(request, (tuple, list)):
            return tuple(request)
        return tuple()

    def _get_request_doc(self, request: Any) -> Any:
        """
        Try to get the original task document.

        For GSM8K, doc usually contains:
            {"question": ..., "answer": ...}

        For MMLU, doc usually contains:
            {"question": ..., "choices": [...], "answer": int}
        """
        if hasattr(request, "doc"):
            return getattr(request, "doc")
        return None

    def _get_request_metadata(self, request: Any) -> Dict[str, Any]:
        """
        Extract task_name, doc_id, and request_idx as defensively as possible.
        """
        task_name = ""
        doc_id = ""
        request_idx = ""
        repeat_idx = ""

        if hasattr(request, "task_name"):
            task_name = str(getattr(request, "task_name"))

        if hasattr(request, "doc_id"):
            doc_id = str(getattr(request, "doc_id"))

        if hasattr(request, "idx"):
            request_idx = str(getattr(request, "idx"))

        metadata = getattr(request, "metadata", None)

        if isinstance(metadata, dict):
            if not task_name and "task_name" in metadata:
                task_name = str(metadata["task_name"])
            if not doc_id and "doc_id" in metadata:
                doc_id = str(metadata["doc_id"])
            if not request_idx and "request_idx" in metadata:
                request_idx = str(metadata["request_idx"])
            if not request_idx and "idx" in metadata:
                request_idx = str(metadata["idx"])
            if "repeat_idx" in metadata:
                repeat_idx = str(metadata["repeat_idx"])

        elif isinstance(metadata, (tuple, list)):
            # Common lm-eval metadata format:
            #   (task_name, doc_id, repeat_idx)
            if len(metadata) > 0 and not task_name:
                task_name = str(metadata[0])
            if len(metadata) > 1 and not doc_id:
                doc_id = str(metadata[1])
            if len(metadata) > 2:
                repeat_idx = str(metadata[2])
            if len(metadata) > 2 and not request_idx:
                request_idx = str(metadata[2])

        elif metadata is not None and not task_name:
            task_name = str(metadata)

        return {
            "task_name": task_name,
            "doc_id": doc_id,
            "request_idx": request_idx,
            "repeat_idx": repeat_idx,
        }

    # -------------------------------------------------------------------------
    # Standard answer extraction
    # -------------------------------------------------------------------------
    def _parse_gsm8k_final_answer(self, answer_text: str) -> str:
        """
        GSM8K reference answer usually looks like:
            reasoning steps...
            #### 460

        Keep the full answer in standard_answer_full and parse the part after
        #### as standard_answer_final.
        """
        if answer_text is None:
            return ""

        answer_text = str(answer_text)
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()

        return ""

    def _choice_index_to_letter(self, idx: int) -> str:
        if 0 <= idx < 26:
            return chr(ord("A") + idx)
        return ""

    def _try_parse_choice_answer(self, answer: Any, choices: Any) -> str:
        """
        Parse MMLU-style answer and return only the answer letter.
        The choice text is intentionally not written to the case CSV.
        """
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            return ""

        answer_idx = None

        # Case 1: answer is integer index, e.g., 0/1/2/3.
        try:
            if isinstance(answer, int):
                answer_idx = int(answer)
            elif isinstance(answer, str) and answer.strip().isdigit():
                answer_idx = int(answer.strip())
        except Exception:
            answer_idx = None

        # Case 2: answer is letter, e.g., A/B/C/D.
        if answer_idx is None and isinstance(answer, str):
            ans = answer.strip().upper()
            if len(ans) == 1 and "A" <= ans <= "Z":
                answer_idx = ord(ans) - ord("A")

        if answer_idx is None:
            return ""

        if not (0 <= answer_idx < len(choices)):
            return ""

        return self._choice_index_to_letter(answer_idx)

    def _extract_standard_answers(self, request: Any) -> Dict[str, str]:
        """
        Extract standard answers from the original lm-eval doc.

        Output fields:
            standard_answer_full:
                Mainly for GSM8K. Full reference reasoning answer.

            standard_answer_final:
                For GSM8K, the final answer after ####.
                For MMLU, the standard answer letter, e.g., A/B/C/D.
        """
        doc = self._get_request_doc(request)

        result = {
            "standard_answer_full": "",
            "standard_answer_final": "",
        }

        if not isinstance(doc, dict):
            return result

        answer = doc.get("answer", "")
        choices = doc.get("choices", None)

        # MMLU-style multiple-choice task.
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            result["standard_answer_final"] = self._try_parse_choice_answer(answer, choices)
            return result

        # GSM8K-style generation task.
        if "answer" in doc:
            answer_full = str(answer)
            answer_final = self._parse_gsm8k_final_answer(answer_full)

            result["standard_answer_full"] = answer_full
            result["standard_answer_final"] = answer_final
            return result

        return result

    def _make_group_key(self, meta: Dict[str, Any], context: Any) -> str:
        """
        For likelihood-based tasks, several choices of the same question usually
        share the same context. We group them to infer the predicted choice.
        Prefer task_name + doc_id + repeat_idx when available; otherwise fall
        back to the context string.
        """
        task_name = str(meta.get("task_name", ""))
        doc_id = str(meta.get("doc_id", ""))
        repeat_idx = str(meta.get("repeat_idx", ""))

        if task_name or doc_id or repeat_idx:
            return self._json_dumps_safe({
                "task_name": task_name,
                "doc_id": doc_id,
                "repeat_idx": repeat_idx,
                "context": str(context),
            })

        return str(context)

    def _safe_float_for_sort(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("-inf")

    def _append_case_row(self, row: List[Any]):
        self._case_log_buffer.append(row)
        if len(self._case_log_buffer) >= 100:
            self._flush_case_log_buffer()

    def _flush_case_log_buffer(self):
        if not getattr(self, "_case_log_buffer", None):
            return

        try:
            with open(self.case_log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(self._case_log_buffer)
            self._case_counter += len(self._case_log_buffer)
            self._case_log_buffer.clear()
            logging.debug("Flushed case output entries to file")
        except Exception as e:
            logging.exception("Failed to flush case output log buffer: %s", e)

    # -------------------------------------------------------------------------
    # Hook and injection logic
    # -------------------------------------------------------------------------
    def _one_time_inject_hook(self, module, inputs):
        for handle in self._hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hook_handles.clear()

        try:
            self._inject_into_parameters()
        except Exception as e:
            logging.exception("Exception during param bitflip injection: %s", e)

        self._injected = True

    def _classify_module(self, param_name: str) -> str:
        name = param_name.lower()
        if any(x in name for x in ("embed", "embedding", "wte", "tok_embeddings")):
            return "embedding"
        if any(x in name for x in ("q_proj", "k_proj", "v_proj", "o_proj", "self_attn", "attn")):
            return "attention"
        if any(x in name for x in ("gate_proj", "up_proj", "down_proj", "fc", "ffn", "mlp")):
            return "ffn"
        if any(x in name for x in ("lm_head", "output", "head")):
            return "output"
        if any(x in name for x in ("norm", "layernorm", "layer_norm", "rmsnorm", "ln")):
            return "layernorm"
        return "other"

    def _inject_into_parameters(self):
        eligible = self._get_eligible_parameters()

        if not eligible:
            logging.warning("No eligible bf16 parameters found for bitflip injection")
            return

        total_bits, total_elements = self._calculate_total_bits_elements(eligible)
        self.total_available_bits = total_bits

        target = "all bf16 parameters" if self.modules is None else f"filtered modules={self.modules}"
        logging.info(
            f"Target scope: {target}; total_elements={total_elements:,}; "
            f"total_available_bits={total_bits:,}"
        )

        flip_count = self._get_flip_count(total_bits, total_elements)

        if flip_count == 0:
            logging.info("No flips sampled (flip_count==0)")
            return

        mapping = self._generate_flip_mapping(eligible, flip_count, total_bits, total_elements)

        per_param_map: Dict[int, List[Tuple[int, int]]] = {}
        for pidx, elem_idx, bpos in mapping:
            per_param_map.setdefault(int(pidx), []).append((int(elem_idx), int(bpos)))

        self._execute_parameter_flips(eligible, per_param_map)

        actual_attempted = sum(len(flips) for flips in per_param_map.values())
        logging.info(
            f"fault_mode={self.fault_mode}; requested_flips={flip_count}; "
            f"attempted_flips={actual_attempted}; logged_actual_flips={self.actual_flips}"
        )

    def _get_eligible_parameters(self) -> List[Tuple[str, torch.Tensor, int, int, str]]:
        """
        Return bf16 parameters selected by the module filter.

        If modules is None, no fine-grained submodule is specified, so the whole
        bf16 parameter space is used as the injection target.
        """
        eligible: List[Tuple[str, torch.Tensor, int, int, str]] = []
        bits = 16
        tag = "bf16"

        for name, param in self.model.named_parameters():
            if param.dtype != torch.bfloat16:
                continue
            if self.modules and not any(k.lower() in name.lower() for k in self.modules):
                continue
            if param.numel() == 0:
                continue

            eligible.append((name, param, param.numel(), bits, tag))

        return eligible

    def _calculate_total_bits_elements(self, eligible: List[Tuple]) -> Tuple[int, int]:
        element_counts = np.array([n for (_, _, n, _, _) in eligible], dtype=np.int64)
        total_elements = int(element_counts.sum())
        total_bits = total_elements * 16
        return total_bits, total_elements

    def _get_flip_count(self, total_bits: int, total_elements: int) -> int:
        """
        Decide the number of bit flips.

        fault_mode=single:
          Always inject exactly one bit flip in the selected target space.

        fault_mode=prob:
          Use Bernoulli/binomial sampling controlled by inject_prob.
          If bit_pos is None, every bit is a candidate.
          If bit_pos is specified, every bf16 element contributes only this bit_pos.
        """
        if self.fault_mode == "single":
            return 1

        if self.bit_pos is None:
            return int(self.np_random_state.binomial(total_bits, self.inject_prob))
        else:
            return int(self.np_random_state.binomial(total_elements, self.inject_prob))

    def _generate_flip_mapping(
        self,
        eligible: List[Tuple],
        flip_count: int,
        total_bits: int,
        total_elements: int,
    ) -> List[Tuple[int, int, int]]:
        if self.bit_pos is None:
            return self._generate_random_bit_mapping(eligible, flip_count, total_bits)
        return self._generate_fixed_bit_mapping(eligible, flip_count, total_elements)

    def _generate_random_bit_mapping(
        self,
        eligible: List[Tuple],
        flip_count: int,
        total_bits: int,
    ) -> List[Tuple[int, int, int]]:
        """Sample bit positions from the selected bf16 parameter bit space."""
        bit_counts = np.array([n * 16 for (_, _, n, _, _) in eligible], dtype=np.int64)
        positions = self.np_random_state.randint(0, total_bits, size=flip_count)
        cum = np.cumsum(bit_counts)
        param_indices = np.searchsorted(cum, positions, side="right")
        prev_cum = np.concatenate(([0], cum[:-1]))
        local_bit_offsets = positions - prev_cum[param_indices]

        mapping: List[Tuple[int, int, int]] = []
        for pi, offset in zip(param_indices, local_bit_offsets):
            elem_idx = int(offset // 16)
            bit_pos = int(offset % 16)
            mapping.append((int(pi), int(elem_idx), int(bit_pos)))

        return mapping
    
    def _generate_fixed_bit_mapping(
        self,
        eligible: List[Tuple],
        flip_count: int,
        total_elements: int,
    ) -> List[Tuple[int, int, int]]:
        """
        Sample bf16 elements, then flip the user-specified bit_pos in each element.

        If fault_mode=single and element_index is provided, directly use that
        element index instead of random sampling.
        """
        elem_counts = np.array([n for (_, _, n, _, _) in eligible], dtype=np.int64)
        cum_e = np.cumsum(elem_counts)

        if self.fault_mode == "single" and self.element_index is not None:
            fixed_elem_index = int(self.element_index)
            param_index = int(np.searchsorted(cum_e, fixed_elem_index, side="right"))
            prev_cum = 0 if param_index == 0 else int(cum_e[param_index - 1])
            local_element_index = int(fixed_elem_index - prev_cum)

            return [(
                param_index,
                local_element_index,
                int(self.bit_pos),
            )]

        elem_choices = self.np_random_state.randint(0, total_elements, size=flip_count)
        param_indices = np.searchsorted(cum_e, elem_choices, side="right")
        prev_cume = np.concatenate(([0], cum_e[:-1]))
        element_indices = elem_choices - prev_cume[param_indices]

        return list(zip(
            param_indices.tolist(),
            element_indices.tolist(),
            [int(self.bit_pos)] * flip_count,
        ))

    def _execute_parameter_flips(self, eligible: List[Tuple], per_param_map: Dict[int, List[Tuple[int, int]]]):
        total_attempted = sum(len(flips) for flips in per_param_map.values())
        logging.info(f"Executing bf16 flips on {len(per_param_map)} parameters, total attempted {total_attempted}")

        for pidx, flips in per_param_map.items():
            if not flips:
                continue

            name, param, _, _, _ = eligible[pidx]
            module_category = self._classify_module(name)

            try:
                self._process_bf16_param(param, name, flips, module_category)
            except Exception as e:
                logging.exception("Failed to flip bits in bf16 param %s: %s", name, e)

    def _process_bf16_param(
        self,
        param: torch.Tensor,
        name: str,
        flips: List[Tuple[int, int]],
        module_category: str,
    ):
        """
        Flip bf16 bits by operating on the high 16 bits of a float32 view.

        PyTorch tensors cannot be directly viewed as bf16 integer buffers through
        NumPy, so we convert bf16 -> fp32, flip bit_pos+16, and cast back to bf16.
        """
        param_cpu = param.detach().cpu().to(torch.float32)
        arr32 = param_cpu.numpy().astype(np.float32)
        flat32 = arr32.reshape(-1)
        int_view32 = flat32.view(np.uint32)

        for elem_idx, bitpos in flips:
            orig_val = float(flat32[elem_idx])
            int_view32[elem_idx] ^= np.uint32(1 << (int(bitpos) + 16))
            new_val = float(flat32[elem_idx])

            if orig_val != new_val:
                self._log_fault(name, param.shape, elem_idx, bitpos, orig_val, new_val, module_category)

        new_tensor = torch.from_numpy(arr32.astype(np.float32)).to(param.device).to(torch.bfloat16)
        param.data.copy_(new_tensor.view(param.data.shape))

    # -------------------------------------------------------------------------
    # Case-level recording for generation-based tasks
    # -------------------------------------------------------------------------
    def generate_until(self, requests):
        """
        Record complete generated outputs.

        Typical request args:
            (context, gen_kwargs)

        This is mainly useful for generation-based tasks such as GSM8K.
        The actual fault injection still happens through the one-time forward
        pre-hook before the first model forward.
        """
        responses = super().generate_until(requests)

        for request, generated_text in zip(requests, responses):
            args = self._get_request_args(request)
            meta = self._get_request_metadata(request)
            std = self._extract_standard_answers(request)

            prompt = args[0] if len(args) > 0 else ""
            gen_kwargs = args[1] if len(args) > 1 else {}

            row = [
                meta["task_name"],
                meta["doc_id"],
                meta["request_idx"],
                str(prompt),
                "",                                      # continuation
                "",                                      # loglikelihood
                "",                                      # predicted_continuation
                str(generated_text),
                self._json_dumps_safe(gen_kwargs),
                std["standard_answer_full"],
                std["standard_answer_final"],
            ]

            self._append_case_row(row)

        self._flush_case_log_buffer()
        return responses

    # -------------------------------------------------------------------------
    # Case-level recording for likelihood-based tasks
    # -------------------------------------------------------------------------
    def loglikelihood(self, requests):
        """
        Record each likelihood request.

        Typical request args:
            (context, continuation)

        This is mainly useful for likelihood-based tasks such as MMLU.
        For MMLU-style multiple choice, lm-eval creates several loglikelihood
        requests for the same question. We group them and infer the predicted
        continuation by max loglikelihood.
        """
        responses = super().loglikelihood(requests)

        parsed_rows: List[Dict[str, Any]] = []
        groups: Dict[str, List[int]] = {}

        for request, result in zip(requests, responses):
            args = self._get_request_args(request)
            meta = self._get_request_metadata(request)
            std = self._extract_standard_answers(request)

            context = args[0] if len(args) > 0 else ""
            continuation = args[1] if len(args) > 1 else ""

            if isinstance(result, (tuple, list)) and len(result) >= 2:
                loglikelihood_value = result[0]
            else:
                loglikelihood_value = result

            group_key = self._make_group_key(meta, context)

            if group_key not in groups:
                groups[group_key] = []

            row_index = len(parsed_rows)
            groups[group_key].append(row_index)

            parsed_rows.append({
                "meta": meta,
                "std": std,
                "context": context,
                "continuation": continuation,
                "loglikelihood": loglikelihood_value,
                "group_key": group_key,
            })

        # Infer best continuation inside each same-question group.
        group_predicted_continuation: Dict[str, str] = {}

        for group_key, row_indices in groups.items():
            best_row_idx = max(
                row_indices,
                key=lambda idx: self._safe_float_for_sort(parsed_rows[idx]["loglikelihood"])
            )
            group_predicted_continuation[group_key] = str(
                parsed_rows[best_row_idx]["continuation"]
            )

        for item in parsed_rows:
            predicted_continuation = group_predicted_continuation[item["group_key"]]

            row = [
                item["meta"]["task_name"],
                item["meta"]["doc_id"],
                item["meta"]["request_idx"],
                str(item["context"]),
                str(item["continuation"]),
                item["loglikelihood"],
                predicted_continuation,
                "",                                      # generated_text
                "",                                      # generation_kwargs
                item["std"]["standard_answer_full"],
                item["std"]["standard_answer_final"],
            ]

            self._append_case_row(row)

        self._flush_case_log_buffer()
        return responses

    def loglikelihood_rolling(self, requests):
        """
        Record rolling loglikelihood requests if the task uses them.

        Typical request args:
            (string,)
        """
        responses = super().loglikelihood_rolling(requests)

        for request, result in zip(requests, responses):
            args = self._get_request_args(request)
            meta = self._get_request_metadata(request)
            std = self._extract_standard_answers(request)

            text = args[0] if len(args) > 0 else ""

            row = [
                meta["task_name"],
                meta["doc_id"],
                meta["request_idx"],
                str(text),
                "",                                      # continuation
                result,                                  # loglikelihood
                "",                                      # predicted_continuation
                "",                                      # generated_text
                "",                                      # generation_kwargs
                std["standard_answer_full"],
                std["standard_answer_final"],
            ]

            self._append_case_row(row)

        self._flush_case_log_buffer()
        return responses

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    def __del__(self):
        try:
            self._flush_log_buffer()
        except Exception:
            pass

        try:
            self._flush_case_log_buffer()
        except Exception:
            pass

        for handle in getattr(self, "_hook_handles", []):
            try:
                handle.remove()
            except Exception:
                pass