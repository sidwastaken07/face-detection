"""Train an FP32 face-recognition student on CASIA-WebFace with a margin loss + teacher distillation.

    python scripts/train_student.py --config configs/student_m.yaml \
        --data <prepared data dir> --teacher-dir <dir with teacher_*.f16.npy> --out /kaggle/working/runs
    # 2 GPUs (Kaggle T4 x2):
    torchrun --nproc_per_node 2 scripts/train_student.py --config ...

Resumes automatically from <out>/<run_name>/last.pt (or --resume <path>, e.g. a previous Kaggle
version's output). Writes metrics.jsonl, best.pt (student weights), results.json.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from fpgaface.config import load_config
from fpgaface.data.casia import CasiaDataset, load_or_scan_labels, identity_subset
from fpgaface.eval.cost import summary
from fpgaface.hw.budget import estimate
from fpgaface.models.students import build_student
from fpgaface.train.augment import TrainTransform
from fpgaface.train.engine import (BenchEvaluator, FaceTrainModule, JsonlLogger, TimeGuard, is_main, param_groups,
                                   save_checkpoint, setup_distributed, unwrap, warmup_cosine)
from fpgaface.utils import load_manifest, save_json, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="prepared data dir (data_manifest.json)")
    ap.add_argument("--teacher-dir", help="dir containing teacher_<name>.f16.npy (needed if kd_weight > 0)")
    ap.add_argument("--out", default="/kaggle/working/runs")
    ap.add_argument("--resume", help="checkpoint to resume from (default: <out>/<run>/last.pt if present)")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=2")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    rank, world, device = setup_distributed()
    main_proc = is_main(rank)
    tc, mc, dc, lc, ec = cfg["train"], cfg["model"], cfg["data"], cfg["loss"], cfg["eval"]
    seed_everything(tc.get("seed", 0) + rank)
    run_dir = os.path.join(args.out, cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    if main_proc:
        save_json(cfg, os.path.join(run_dir, "config.json"))
    log = JsonlLogger(os.path.join(run_dir, "metrics.jsonl"), main_proc)
    guard = TimeGuard(tc.get("max_hours", 11.0))

    # ---------------------------------------------------------------- data
    manifest = load_manifest(args.data)
    casia = manifest["casia"]
    _, all_labels = load_or_scan_labels(casia["rec"], casia["labels"])
    indices = identity_subset(all_labels, dc["num_ids"], dc.get("subset_seed", 0)) if dc.get("num_ids") else None
    ds = CasiaDataset(casia["rec"], casia["labels"], TrainTransform(mc["input_size"], **dc.get("aug", {})), indices)
    sampler = DistributedSampler(ds, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=tc["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=dc.get("workers", 4), pin_memory=device.type == "cuda", drop_last=True,
                        persistent_workers=dc.get("workers", 4) > 0)
    teacher_emb, teacher_dim = None, 512
    if lc.get("kd") and float(lc.get("kd_weight", 0)) > 0:
        path = os.path.join(args.teacher_dir, f"teacher_{lc.get('teacher', 'arcface_r50')}.f16.npy")
        teacher_emb = torch.from_numpy(np.load(path))  # [N_total, 512] fp16 in RAM, indexed by dataset index
        teacher_dim = teacher_emb.shape[1]
    evaluator = BenchEvaluator(manifest["bins"], ec["bins"], mc["input_size"]) if main_proc else None
    if main_proc:
        print(f"{cfg['run_name']}: {len(ds)} images, {ds.num_classes} identities, world={world}, "
              f"{len(loader)} steps/epoch", flush=True)

    # ---------------------------------------------------------------- model
    student = build_student(mc["arch"], width=mc["width"], emb_dim=mc["emb_dim"], input_size=mc["input_size"])
    module = FaceTrainModule(student, mc["emb_dim"], ds.num_classes, lc, teacher_dim).to(device)
    if tc.get("channels_last", True):
        module = module.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(param_groups(module, tc["weight_decay"]), lr=tc["lr"], momentum=tc.get("momentum", 0.9),
                          nesterov=True)
    steps_per_epoch = len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(tc["epochs"] * steps_per_epoch, int(tc.get("warmup_epochs", 1) * steps_per_epoch)))
    amp = tc.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best = 0, -1.0
    resume = args.resume or (os.path.join(run_dir, "last.pt") if os.path.exists(os.path.join(run_dir, "last.pt")) else None)
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        module.load_state_dict(ck["module"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"], ck.get("best", -1.0)
        if main_proc:
            print(f"resumed from {resume} at epoch {start_epoch}", flush=True)
            if resume != os.path.join(run_dir, "last.pt") and os.path.exists(os.path.join(os.path.dirname(resume), "best.pt")):
                shutil.copy(os.path.join(os.path.dirname(resume), "best.pt"), os.path.join(run_dir, "best.pt"))
    if world > 1:
        module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device.index])

    def checkpoint(epoch):
        if main_proc:
            save_checkpoint(os.path.join(run_dir, "last.pt"), module=unwrap(module).state_dict(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best=best, cfg=cfg)

    # ---------------------------------------------------------------- train
    epoch_time = None
    stopped_early = False
    for epoch in range(start_epoch, tc["epochs"]):
        if epoch_time and not guard.can_fit(epoch_time):
            stopped_early = True
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        module.train()
        t0 = t_log = time.time()
        seen = 0
        for step, (x, y, idx) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if tc.get("channels_last", True):
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            t = teacher_emb[idx].to(device, non_blocking=True) if teacher_emb is not None else None
            with torch.autocast(device_type=device.type, enabled=amp):
                loss, ce, kd = module(x, y, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += len(x) * world
            if main_proc and step % tc.get("log_every", 100) == 0:
                rate = seen / (time.time() - t0)
                print(f"ep {epoch} {step}/{steps_per_epoch} loss {loss.item():.3f} ce {ce.item():.3f} "
                      f"kd {kd.item():.4f} lr {sched.get_last_lr()[0]:.4f} {rate:.0f} img/s", flush=True)
                log(epoch=epoch, step=step, loss=loss.item(), ce=ce.item(), kd=kd.item(),
                    lr=sched.get_last_lr()[0], img_s=rate, elapsed_h=guard.elapsed() / 3600)
            if guard.expired():
                stopped_early = True
                break
        if stopped_early:  # partial epoch is discarded; resume restarts it
            break
        epoch_time = time.time() - t0

        if main_proc:
            res = evaluator(unwrap(module).student, device, amp) if (epoch + 1) % ec.get("every", 1) == 0 or epoch + 1 == tc["epochs"] else {}
            accs = {k: v["acc"] for k, v in res.items() if isinstance(v, dict)}
            print(f"== epoch {epoch} done in {epoch_time / 60:.1f} min | " +
                  " ".join(f"{k} {v * 100:.2f}" for k, v in accs.items()), flush=True)
            log(epoch=epoch, epoch_min=epoch_time / 60, **{f"val_{k}": v for k, v in accs.items()},
                val_mean=res.get("mean_acc"))
            if res and res["mean_acc"] > best:
                best = res["mean_acc"]
                torch.save(unwrap(module).student.state_dict(), os.path.join(run_dir, "best.pt"))
        checkpoint(epoch + 1)
        if world > 1:
            torch.distributed.barrier()

    if stopped_early:
        checkpoint(epoch)
        if main_proc:
            print(f"time limit reached — stopped before epoch {epoch}; re-run to resume.", flush=True)
        return

    # ---------------------------------------------------------------- final evaluation
    if main_proc:
        student = unwrap(module).student
        if os.path.exists(os.path.join(run_dir, "best.pt")):
            student.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
        final = BenchEvaluator(manifest["bins"], ec.get("final_bins", ec["bins"]), mc["input_size"])(student, device, amp)
        cpu_model = build_student(mc["arch"], width=mc["width"], emb_dim=mc["emb_dim"], input_size=mc["input_size"])
        results = {
            "run_name": cfg["run_name"],
            "verification": final,
            "cost": summary(cpu_model, (1, 3, mc["input_size"], mc["input_size"])),
            "budget_w4a4_z7020": estimate(cpu_model, mc["input_size"], 4, 4, "z7020"),
            "train_hours": guard.elapsed() / 3600,
            "epochs": tc["epochs"],
        }
        save_json(results, os.path.join(run_dir, "results.json"))
        print("final:", {k: round(v["acc"] * 100, 2) for k, v in final.items() if isinstance(v, dict)}, flush=True)


if __name__ == "__main__":
    main()
