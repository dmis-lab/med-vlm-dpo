import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from .llavamed_mDPO_dpo_trainer import LlavaMedDPOTrainer
from .llavamed_copo_trainer import LlavaMedCOPOTrainer

class LlavaMedmDPOTrainer(LlavaMedDPOTrainer, LlavaMedCOPOTrainer):
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Combines DPO and COPO losses with SFT loss to compute the final loss.
        """
        
        # Compute DPO loss and metrics
        dpo_loss, ancpo_loss, dpo_metrics = LlavaMedDPOTrainer.get_batch_metrics(self, inputs, train_eval="train")

        # Compute COPO loss and metrics
        copo_loss, copo_metrics = LlavaMedCOPOTrainer.get_batch_metrics(self, inputs, train_eval="train")

        # Final loss
        final_loss = dpo_loss + copo_loss + ancpo_loss

        # Add prefixes to metrics to distinguish DPO and COPO metrics
        dpo_metrics_prefixed = {f"dpo_{key}": value for key, value in dpo_metrics.items()}
        copo_metrics_prefixed = {f"copo_{key}": value for key, value in copo_metrics.items()}

        # Merge metrics with prefixes
        combined_metrics = {**dpo_metrics_prefixed, **copo_metrics_prefixed}
        combined_metrics["ancpo_loss"] = ancpo_loss.detach().cpu().item()

        # Force log metrics
        if self.accelerator.is_main_process:
            self.store_metrics(combined_metrics, train_eval="train")
            
        if return_outputs:
            return final_loss, combined_metrics
        return final_loss
