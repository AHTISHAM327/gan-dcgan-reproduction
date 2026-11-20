"""Sanity checks from the course's PyTorch checklist. Owner: Ahtisham."""
import math
import torch
from src.losses import d_loss_fn, g_loss_fn


def first_loss_check(G, D, real, z):
    """At the start D cannot tell real from fake, so D(x) is about 0.5
    and each part of the D loss is about ln 2 = 0.693 (C = 2 classes: real, fake)."""
    G.eval(); D.eval()
    with torch.no_grad():
        real_logits, fake_logits = D(real), D(G(z))
        total, real_part, fake_part = d_loss_fn(real_logits, fake_logits)
        g_loss = g_loss_fn(fake_logits)
    print(f'ln 2 = {math.log(2):.4f}')
    print(f'D loss on real = {real_part.item():.4f}  | D loss on fake = {fake_part.item():.4f}  | total = {total.item():.4f} (ln 4 = {math.log(4):.4f})')
    print(f'G loss = {g_loss.item():.4f}  | mean D(x) = {torch.sigmoid(real_logits).mean():.4f}  | mean D(G(z)) = {torch.sigmoid(fake_logits).mean():.4f}')
    ok = abs(real_part.item() - math.log(2)) < 0.05 and abs(fake_part.item() - math.log(2)) < 0.05
    print('OK: first loss is close to ln 2' if ok else 'CHECK: first loss is far from ln 2 (look at the initialisation)')
    return ok


def gradient_check(G, D, real, z):
    """One backward pass each for D and G; every layer with parameters must get a non-zero gradient."""
    G.train(); D.train()
    G.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)
    d_total, _, _ = d_loss_fn(D(real), D(G(z).detach()))
    d_total.backward()
    g_loss = g_loss_fn(D(G(z)))
    G.zero_grad(set_to_none=True)
    g_loss.backward()                       # gives G its gradients (D's are refilled too; we only read G's here)
    all_ok = True
    for model_name, model in (('D', D), ('G', G)):
        for name, p in model.named_parameters():
            norm = 0.0 if p.grad is None else p.grad.norm().item()
            flag = 'OK' if norm > 0 else 'ZERO!'
            all_ok &= norm > 0
            print(f'{model_name}.{name:<22} grad norm = {norm:.3e}  {flag}')
    print('OK: every layer receives a non-zero gradient' if all_ok else 'PROBLEM: some layer gets no gradient')
    G.zero_grad(set_to_none=True); D.zero_grad(set_to_none=True)
    return all_ok
