import os
import csv
import json
import logging
import tempfile
from datetime import datetime
from typing import Optional, Any, Dict, List, Tuple

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("hf_record")
class RecordOnlyHFLM(HFLM):
    """
    Clean HFLM wrapper for case-level output recording.

    This model does NOT perform any fault injection.

    It records:
      1. generation-based tasks, e.g., GSM8K:
         prompt, generated_text, standard_answer_full, standard_answer_final

      2. likelihood-based tasks, e.g., MMLU:
         prompt, continuation, loglikelihood, predicted_continuation,
         standard_answer_choice_letter, standard_answer_choice_text
    """

    def __init__(
        self,
        pretrained: str,
        output_path: Optional[str] = None,
        case_output_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)

        if output_path:
            csv_dir = os.path.dirname(output_path)
            os.makedirs(csv_dir, exist_ok=True)
            logging.info(f"Using output directory from command line: {csv_dir}")
        else:
            csv_dir = "./results/hf_record"
            os.makedirs(csv_dir, exist_ok=True)
            logging.warning(f"No output_path provided, using default directory: {csv_dir}")

        self.case_log_file = (
            case_output_path
            if case_output_path is not None
            else self._create_case_log_file(csv_dir)
        )

        if case_output_path is not None:
            self._init_case_log_file(case_output_path)

        self._case_log_buffer: List[List[Any]] = []
        self._init_logging()

        logging.info(
            "Initialized clean case recorder "
            f"(pretrained={pretrained}, output_dir={csv_dir}, "
            f"case_log_file={self.case_log_file})"
        )

    # -------------------------------------------------------------------------
    # Basic helpers
    # -------------------------------------------------------------------------
    def _init_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

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

        In most lm-eval versions, Instance contains .doc.
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

        elif isinstance(metadata, (tuple, list)):
            # Common lm-eval metadata format:
            #   (task_name, doc_id, repeat_idx)
            if len(metadata) > 0 and not task_name:
                task_name = str(metadata[0])
            if len(metadata) > 1 and not doc_id:
                doc_id = str(metadata[1])
            if len(metadata) > 2 and not request_idx:
                request_idx = str(metadata[2])

        elif metadata is not None and not task_name:
            task_name = str(metadata)

        return {
            "task_name": task_name,
            "doc_id": doc_id,
            "request_idx": request_idx,
        }

    # -------------------------------------------------------------------------
    # Standard answer extraction
    # -------------------------------------------------------------------------
    def _parse_gsm8k_final_answer(self, answer_text: str) -> str:
        """
        GSM8K reference answer usually looks like:
            reasoning steps...
            #### 460

        We keep the full answer in standard_answer_full and parse the part
        after #### as standard_answer_final.
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

    def _try_parse_choice_answer(
        self,
        answer: Any,
        choices: Any,
    ) -> Tuple[str, str]:
        """
        Parse MMLU-style answer.

        Returns:
            standard_answer_choice_letter,
            standard_answer_choice_text
        """
        if not isinstance(choices, (list, tuple)) or len(choices) == 0:
            return "", ""

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
            return "", ""

        if not (0 <= answer_idx < len(choices)):
            return "", ""

        answer_letter = self._choice_index_to_letter(answer_idx)
        answer_text = str(choices[answer_idx])

        return answer_letter, answer_text

    def _extract_standard_answers(self, request: Any) -> Dict[str, str]:
        """
        Extract standard answers from the original lm-eval doc.

        Output fields:
            standard_answer_choice_letter:
                Mainly for MMLU. Example: "B"

            standard_answer_choice_text:
                Mainly for MMLU. Example: "mitochondria"

            standard_answer_full:
                Mainly for GSM8K. Full reference reasoning answer.

            standard_answer_final:
                Mainly for GSM8K. The final answer after ####.
                For MMLU, this is set to the answer letter when available.
        """
        doc = self._get_request_doc(request)

        result = {
            "standard_answer_choice_letter": "",
            "standard_answer_choice_text": "",
            "standard_answer_full": "",
            "standard_answer_final": "",
        }

        if not isinstance(doc, dict):
            return result

        answer = doc.get("answer", "")
        choices = doc.get("choices", None)

        # MMLU-style multiple-choice task.
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            answer_letter, answer_text = self._try_parse_choice_answer(answer, choices)

            result["standard_answer_choice_letter"] = answer_letter
            result["standard_answer_choice_text"] = answer_text
            result["standard_answer_final"] = answer_letter

            return result

        # GSM8K-style generation task.
        if "answer" in doc:
            answer_full = str(answer)
            answer_final = self._parse_gsm8k_final_answer(answer_full)

            result["standard_answer_full"] = answer_full
            result["standard_answer_final"] = answer_final

            return result

        return result

    # -------------------------------------------------------------------------
    # CSV creation and writing
    # -------------------------------------------------------------------------
    def _case_header(self) -> List[str]:
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
            filename = f"case_outputs_clean_{timestamp}.csv"
            filepath = os.path.join(directory, filename)
            self._init_case_log_file(filepath)
            logging.info(f"Created clean case output log: {filepath}")
            return filepath

        except Exception as e:
            logging.error(f"Failed to create case log file in {directory}: {e}")
            temp_file = tempfile.mktemp(suffix=".csv", dir=directory)
            self._init_case_log_file(temp_file)
            logging.warning(f"Using temporary case log file: {temp_file}")
            return temp_file

    def _init_case_log_file(self, filepath: str):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self._case_header())

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
            self._case_log_buffer.clear()
        except Exception as e:
            logging.exception("Failed to flush case log buffer: %s", e)

    # -------------------------------------------------------------------------
    # Recording logic for generation-based tasks
    # -------------------------------------------------------------------------
    def generate_until(self, requests):
        """
        Record complete generated outputs.

        Typical request args:
            (context, gen_kwargs)

        Mainly useful for GSM8K.
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
                "",
                str(generated_text),
                self._json_dumps_safe(gen_kwargs),
                std["standard_answer_full"],
                std["standard_answer_final"],
            ]

            self._append_case_row(row)

        self._flush_case_log_buffer()
        return responses

    # -------------------------------------------------------------------------
    # Recording logic for likelihood-based tasks
    # -------------------------------------------------------------------------
    def loglikelihood(self, requests):
        """
        Record each likelihood request.

        Typical request args:
            (context, continuation)

        Mainly useful for MMLU.

        For MMLU-style multiple choice, lm-eval creates several loglikelihood
        requests with the same prompt but different continuations. We group them
        together and infer predicted_continuation by max loglikelihood.
        """
        responses = super().loglikelihood(requests)

        parsed_rows: List[Dict[str, Any]] = []
        groups: Dict[Tuple[str, str, str], List[int]] = {}

        for request, result in zip(requests, responses):
            args = self._get_request_args(request)
            meta = self._get_request_metadata(request)
            std = self._extract_standard_answers(request)

            prompt = args[0] if len(args) > 0 else ""
            continuation = args[1] if len(args) > 1 else ""

            if isinstance(result, (tuple, list)) and len(result) >= 2:
                loglikelihood_value = result[0]
            else:
                loglikelihood_value = result

            group_key = (
                str(meta["task_name"]),
                str(meta["doc_id"]),
                str(prompt),
            )

            if group_key not in groups:
                groups[group_key] = []

            row_idx = len(parsed_rows)
            groups[group_key].append(row_idx)

            parsed_rows.append({
                "meta": meta,
                "std": std,
                "prompt": prompt,
                "continuation": continuation,
                "loglikelihood": loglikelihood_value,
                "group_key": group_key,
            })

        # Infer predicted continuation for each same-prompt group.
        group_predicted_continuation: Dict[Tuple[str, str, str], str] = {}

        for group_key, row_indices in groups.items():
            best_row_idx = max(
                row_indices,
                key=lambda idx: float(parsed_rows[idx]["loglikelihood"])
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
                str(item["prompt"]),
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
        Record rolling loglikelihood requests if a task uses them.

        Typical request args:
            (string,)
        """
        responses = super().loglikelihood_rolling(requests)

        for request, result in zip(requests, responses):
            args = self._get_request_args(request)
            meta = self._get_request_metadata(request)
            std = self._extract_standard_answers(request)

            prompt = args[0] if len(args) > 0 else ""

            row = [
                meta["task_name"],
                meta["doc_id"],
                meta["request_idx"],
                str(prompt),
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
            self._flush_case_log_buffer()
        except Exception:
            pass