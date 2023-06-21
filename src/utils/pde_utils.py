import torch


def make_mesh(x: torch.tensor, t: torch.tensor, stack_output: bool = False) -> torch.tensor:
    """
    create 2d coordinate
    x : (Nx,)
    y : (Nt,)
    return: (Nx x Nt, 2)
    """
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    xx = xx.reshape(-1, 1)
    tt = tt.reshape(-1, 1)
    if stack_output:
        return torch.cat([xx, tt], dim=1)
    else:
        return xx, tt