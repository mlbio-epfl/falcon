import torch

def gather_from_all(payload):
    all_payload = [torch.zeros_like(payload) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_payload, payload)
    all_payload[torch.distributed.get_rank()] = payload
    all_payload = torch.cat(all_payload, dim=0)
    return all_payload

def gather_from_all_without_reduce(payload):
    all_payload = [torch.zeros_like(payload) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(all_payload, payload)
    all_payload[torch.distributed.get_rank()] = payload
    return all_payload
