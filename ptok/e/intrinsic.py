from dataclasses import dataclass
import torch
from tqdm import tqdm


@dataclass
class MLMEvaluationResult:
    loss: float
    top1_accuracy: float
    topk_accuracy: float
    masked_tokens: int



class MLMAccuracyEvaluator:
    """
    Masked LM accuracy evaluator.
    """

    def __init__(self, model, device="cpu", top_k=5):
        self.model = model.to(device)
        self.device = device
        self.top_k = top_k

    @torch.no_grad()
    def evaluate(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        batches = 0

        top1_correct = 0
        topk_correct = 0
        masked_tokens = 0

        progress = tqdm(dataloader, desc="Evaluating", unit="batch")

        for batch in progress:
            # Move to device
            #
            batch = { k: v.to(self.device) for k, v in batch.items() }
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            batches += 1
            logits = outputs.logits
            labels = batch["labels"]

            # Only evaluate masked positions
            mask = labels != -100

            if mask.sum() == 0:
                continue

            masked_tokens += mask.sum().item()

            #
            # --------
            # Top-1
            # --------
            #
            pred = logits.argmax(dim=-1)

            top1_correct += (pred[mask] == labels[mask]).sum().item()

            #
            # --------
            # Top-k
            # --------
            #
            topk = logits.topk(self.top_k, dim=-1).indices

            #
            # Shape:
            #
            # (num_masked, k)
            #
            topk = topk[mask]

            gold = labels[mask].unsqueeze(-1)

            topk_correct += (topk == gold).any(dim=-1).sum().item()

            progress.set_postfix(
                loss=f"{total_loss / batches:.4f}",
                top1=f"{100 * top1_correct / masked_tokens:.2f}%",
                topk=f"{100 * topk_correct / masked_tokens:.2f}%"
            )

        return MLMEvaluationResult(
            loss=total_loss / batches,
            top1_accuracy=top1_correct / masked_tokens,
            topk_accuracy=topk_correct / masked_tokens,
            masked_tokens=masked_tokens
        )


