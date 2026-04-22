"""FedAvgServer — cloud server that aggregates local updates."""
import io, copy, time
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMTabular
import config

class FedAvgServer:
    def __init__(self):
        self.model = LSTMTabular(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS)
        for name, p in self.model.named_parameters():
            if "weight" in name and p.dim() >= 2:
                torch.nn.init.xavier_uniform_(p)
            elif "bias" in name:
                torch.nn.init.zeros_(p)
        self.global_state_dict = copy.deepcopy(self.model.state_dict())

    def get_global_state_dict(self):
        return copy.deepcopy(self.global_state_dict)

    def aggregate(self, updates):
        """Weighted FedAvg. updates = [(state_dict, n_samples), ...]"""
        t0 = time.perf_counter()
        total = sum(n for _, n in updates)
        weights = [n / total for _, n in updates]
        avg = {}
        for key in updates[0][0].keys():
            v0 = updates[0][0][key]
            if not torch.is_floating_point(v0):
                avg[key] = v0.clone(); continue
            acc = torch.zeros_like(v0, dtype=torch.float32)
            for (sd, _), w in zip(updates, weights):
                acc += w * sd[key].float()
            avg[key] = acc.to(dtype=v0.dtype)
        self.global_state_dict = avg
        agg_sec = time.perf_counter() - t0
        buf = io.BytesIO(); torch.save(avg, buf)
        return avg, agg_sec, buf.tell()

    def save(self, path):
        torch.save(self.global_state_dict, path)
        print(f"[SERVER] Global model saved -> {path}")
