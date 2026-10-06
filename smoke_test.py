import gc, time, torch, torch.nn as nn
from peft import LoraConfig, get_peft_model
import fm_registry as R

def smoke_test(names, x, y, num_classes, device, class_weights=None):
    """Per model: load, embed real images, attach LoRA, check init equality, run 1 train step."""
    report = []
    for name in names:
        t0, row = time.time(), {'model': name}
        try:
            bb = R.load_backbone(name, device)
            with torch.no_grad():
                e = bb(x.to(device))
            row['dim'] = e.shape[-1]
            assert e.shape[-1] == bb.spec['dim'], f"dim {e.shape[-1]} != spec {bb.spec['dim']}"
            assert torch.isfinite(e).all(), "non-finite embeddings"

            lora_bb = get_peft_model(bb, LoraConfig(   # in place: saves memory, model is discarded after
                r=8, lora_alpha=16, lora_dropout=0.0, target_modules=bb.spec['lora_targets'], bias='none'))
            n_lora = sum(p.numel() for n, p in lora_bb.named_parameters() if p.requires_grad)
            row['lora_params'] = n_lora
            assert n_lora > 0, "LoRA attached to nothing"
            lora_bb.eval()
            with torch.no_grad():
                same = torch.allclose(lora_bb(x.to(device)), e, atol=1e-4)
            assert same, "LoRA at init changes output"

            head = nn.Linear(row['dim'], num_classes).to(device)
            params = [p for p in lora_bb.parameters() if p.requires_grad] + list(head.parameters())
            opt = torch.optim.AdamW(params, lr=1e-4)
            lora_bb.train()
            loss = nn.CrossEntropyLoss(weight=None if class_weights is None else class_weights.to(device))(
                head(lora_bb(x.to(device))), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            assert torch.isfinite(loss), "non-finite loss"
            row.update(loss=round(loss.item(), 3), status='OK')
            del lora_bb, head, opt
        except Exception as ex:
            row.update(status=f'FAIL: {type(ex).__name__}: {str(ex)[:150]}')
        row['sec'] = round(time.time() - t0, 1)
        report.append(row); print(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report
