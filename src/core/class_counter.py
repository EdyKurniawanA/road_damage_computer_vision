# class_counter.py
from collections import Counter
import torch


class ClassCounter:
    """
    Tracks YOLO detections per frame.
    Supports YOLOv5 or YOLOv8/11 outputs.
    """

    def __init__(self, accumulate: bool = False):
        self.accumulate = accumulate
        self.counter = Counter()

    def update_from_yolo(self, results):
        """
        Update the counter using YOLO results object.
        Handles YOLOv5 and YOLOv8/11.
        """

        if results is None:
            return

        # ---- YOLOv5 (results.xyxy, results.names) ----
        if hasattr(results, "xyxy"):
            names = results.names
            detections = results.xyxy[0]  # (N, 6): [x1, y1, x2, y2, conf, cls]
            cls_ids = detections[:, -1].cpu().numpy().astype(int)
            labels = [names[int(c)] for c in cls_ids]

        # ---- YOLOv8/11 (results[0].boxes.cls, results[0].names) ----
        elif isinstance(results, list) and hasattr(results[0], "boxes"):
            names = results[0].names
            classes = results[0].boxes.cls
            if isinstance(classes, torch.Tensor):
                classes = classes.cpu().numpy().astype(int)
            labels = [names[int(c)] for c in classes]

        else:
            return  # unsupported format

        if not self.accumulate:
            self.counter = Counter()  # reset per frame

        self.counter.update(labels)

    def get_counts(self):
        """Return dict with class counts, e.g. {"person": 3, "car": 1}"""
        return dict(self.counter)

    def reset(self):
        """Manually reset counts (useful if accumulate=True)."""
        self.counter = Counter()

    def update_from_roboflow(self, result):
        """
        Update the counter using Roboflow Cloud API result.
        Handles various Roboflow prediction formats.
        """
        if result is None:
            return
            
        try:
            # 1) Prefer explicit class counts if the workflow provides them (e.g., "count_objects")
            counts = self._extract_counts_from_roboflow_result(result)
            if counts:
                if not self.accumulate:
                    self.counter = Counter()
                # Normalize values to int
                normalized = {str(k): int(v) for k, v in counts.items() if self._is_number(v)}
                if normalized:
                    self.counter.update(normalized)
                    return

            # 2) Otherwise, extract predictions and derive labels
            predictions = self._extract_roboflow_predictions(result)

            labels = []
            if predictions:
                # Extract labels from flat predictions list
                for pred in predictions:
                    if isinstance(pred, dict):
                        # Try multiple possible label field names
                        label = (
                            pred.get("class")
                            or pred.get("class_name")
                            or pred.get("label")
                            or pred.get("name")
                            or pred.get("category")
                            or "object"
                        )
                        if label and label != "object":  # Only count valid labels
                            labels.append(label)

            # 3) Fallback: recursively search for labels anywhere in the result
            if not labels:
                labels = self._collect_labels_recursive(result)

            if labels:
                if not self.accumulate:
                    self.counter = Counter()  # reset per frame
                self.counter.update(labels)

        except Exception:
            # Fail silently to avoid affecting runtime
            pass

    def _extract_roboflow_predictions(self, result):
        """Extract predictions from Roboflow result dictionary"""
        if not isinstance(result, dict):
            return []
            
        # Try different possible locations for predictions
        predictions = None
        
        # Check for model_predictions.predictions
        if ("model_predictions" in result and 
            isinstance(result["model_predictions"], dict) and 
            "predictions" in result["model_predictions"]):
            predictions = result["model_predictions"]["predictions"]
            
        # Check for output.predictions/detections/results
        elif ("output" in result and 
              isinstance(result["output"], dict)):
            out = result["output"]
            if isinstance(out, dict):
                predictions = (out.get("predictions") or 
                             out.get("detections") or 
                             out.get("results"))
                             
        # Check for direct predictions/detections
        elif "predictions" in result:
            predictions = result["predictions"]
        elif "detections" in result:
            predictions = result["detections"]
        elif "results" in result and isinstance(result["results"], list):
            predictions = result["results"]
            
        return predictions if isinstance(predictions, list) else []

    def _collect_labels_recursive(self, obj):
        """
        Recursively collect label-like strings from nested dict/list structures.
        Looks for common keys used by Roboflow: class, class_name, label, name, category.
        """
        collected = []

        def _walk(node):
            try:
                if isinstance(node, dict):
                    # If dict has a label-like key, collect it
                    for key in ("class", "class_name", "label", "name", "category"):
                        if key in node and isinstance(node[key], str):
                            val = node[key].strip()
                            if val:
                                collected.append(val)
                    # Recurse into values
                    for v in node.values():
                        _walk(v)
                elif isinstance(node, list):
                    for item in node:
                        _walk(item)
                # Ignore primitives
            except Exception:
                # Be resilient to unexpected structures
                pass

        _walk(obj)

        return collected

    def _extract_counts_from_roboflow_result(self, result):
        """
        Try to extract a per-class counts dictionary from a Roboflow Workflow result.
        Supports outputs like:
          result["output"]["count_objects"]["output"] -> {"class": count, ...}
          result["count_objects"]["output"] -> {...}
          result["count_objects"] -> {...}
        Falls back to the first dict found whose values are all numeric.
        """
        try:
            # Common explicit paths
            if isinstance(result, dict):
                out = result.get("output")
                if isinstance(out, dict):
                    co = out.get("count_objects")
                    if isinstance(co, dict):
                        # Unwrap nested {"output": {...}} if present
                        if isinstance(co.get("output"), dict):
                            return co["output"]
                        # Or counts may already be a dict of class->number
                        if self._looks_like_counts_dict(co):
                            return co
                # Root-level variants
                co_root = result.get("count_objects")
                if isinstance(co_root, dict):
                    if isinstance(co_root.get("output"), dict):
                        return co_root["output"]
                    if self._looks_like_counts_dict(co_root):
                        return co_root

            # Recursive fallback: search for any dict with string keys and numeric values
            found = self._find_counts_dict_recursive(result)
            if isinstance(found, dict):
                return found
        except Exception:
            pass
        return {}

    def _find_counts_dict_recursive(self, obj):
        """Depth-first search for a dict that looks like {str: number}.*"""
        try:
            if isinstance(obj, dict):
                if self._looks_like_counts_dict(obj):
                    return obj
                for v in obj.values():
                    found = self._find_counts_dict_recursive(v)
                    if isinstance(found, dict):
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = self._find_counts_dict_recursive(item)
                    if isinstance(found, dict):
                        return found
        except Exception:
            pass
        return None

    @staticmethod
    def _looks_like_counts_dict(obj):
        if not isinstance(obj, dict):
            return False
        if not obj:
            return False
        for k, v in obj.items():
            if not isinstance(k, str):
                return False
            if not ClassCounter._is_number(v):
                return False
        return True

    @staticmethod
    def _is_number(value):
        try:
            float(value)
            return True
        except Exception:
            return False

    def update_from_labels(self, labels):
        """Update the counter using a list of class label strings."""
        if labels is None:
            return
        try:
            if not self.accumulate:
                self.counter = Counter()
            # Ensure labels is a flat iterable of strings
            label_list = []
            for label in labels:
                if isinstance(label, str):
                    label_list.append(label)
            if label_list:
                self.counter.update(label_list)
        except Exception:
            # Fail silently to avoid affecting runtime
            pass