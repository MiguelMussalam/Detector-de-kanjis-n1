"""
train.py
========
Loop de treino do classificador de kanji (ResNet-18 sobre crops sintéticos).

Roda localmente (CPU) ou no Kaggle (GPU). Local é viável só para testes rápidos
com poucas épocas/classes — o treino real (30 épocas, 1232 classes) deve
rodar em GPU (Kaggle T4), senão o tempo por época fica proibitivo.

Uso:
    python -m src.classifier.train
"""

import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    CLF_EPOCHS, CLF_LR, CLF_WEIGHT_DECAY, CLF_PATIENCE,
    CLF_PROJECT_NAME, CLF_BATCH_SIZE,
    KAGGLE_WORKERS, LOCAL_WORKERS, ROOT_DIR,
    CLF_FINETUNE_FROM, CLF_FINETUNE_LR,
)
from src.classifier.dataset import build_datasets, build_dataloaders
from src.classifier.model import build_model, count_parameters


IS_KAGGLE = (
    os.path.exists("/kaggle/input") or
    "KAGGLE_DATA_PROXY_TOKEN" in os.environ or
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Uma época (treino ou avaliação)
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer=None):
    """
    Roda uma passada completa pelo loader.
    Se `optimizer` for fornecido, faz backward+step (modo treino).
    Caso contrário, só avalia (modo val) — sem gradiente.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += imgs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

def main():
    print(f"[INFO] Device: {DEVICE}")
    if not torch.cuda.is_available():
        print("[AVISO] Rodando em CPU — treino completo (1232 classes) vai ser lento aqui. "
              "Use --limit no generate_crops + KD_CLF_EPOCHS baixo para testar, "
              "e rode o treino de verdade no Kaggle (GPU).")

    print("[INFO] Carregando datasets...")
    train_ds, val_ds = build_datasets()
    num_workers = KAGGLE_WORKERS if IS_KAGGLE else LOCAL_WORKERS
    train_loader, val_loader = build_dataloaders(
        train_ds, val_ds, batch_size=CLF_BATCH_SIZE, num_workers=num_workers
    )
    num_classes = len(train_ds.classes)
    print(f"[INFO] {len(train_ds)} treino / {len(val_ds)} val / {num_classes} classes")

    print("[INFO] Construindo modelo...")
    model = build_model(num_classes=num_classes).to(DEVICE)

    lr = CLF_LR
    if CLF_FINETUNE_FROM:
        print(f"[INFO] Fine-tuning a partir de: {CLF_FINETUNE_FROM}")
        ckpt = torch.load(CLF_FINETUNE_FROM, map_location=DEVICE, weights_only=False)
        if ckpt["classes"] != train_ds.classes:
            raise RuntimeError(
                f"Classes do checkpoint de fine-tuning ({len(ckpt['classes'])}) nao batem "
                f"com as classes do dataset atual ({len(train_ds.classes)}). Carregar os "
                f"pesos assim corromperia o alinhamento indice->classe. Confira se o dataset "
                f"(sintetico + real mesclado) tem exatamente as mesmas classes do checkpoint."
            )
        model.load_state_dict(ckpt["model_state"])
        lr = CLF_FINETUNE_LR
        print(f"[INFO] LR de fine-tuning: {lr} (treino do zero usaria {CLF_LR})")

    print(f"[INFO] Parametros treinaveis: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=CLF_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CLF_EPOCHS)

    run_dir = os.path.join(ROOT_DIR, CLF_PROJECT_NAME, "run")
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "results.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    best_val_acc = 0.0
    epochs_sem_melhora = 0

    print(f"[INFO] Treinando por ate {CLF_EPOCHS} epocas (patience={CLF_PATIENCE})")
    for epoch in range(1, CLF_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()

        dt = time.time() - t0
        lr_atual = scheduler.get_last_lr()[0]
        print(
            f"[{epoch:03d}/{CLF_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr_atual:.2e} | {dt:.1f}s"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr_atual:.8f}\n")

        checkpoint = {
            "model_state": model.state_dict(),
            "classes": train_ds.classes,
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(checkpoint, last_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_sem_melhora = 0
            torch.save(checkpoint, best_path)
            print(f"          -> novo melhor val_acc: {val_acc:.4f} (salvo em {best_path})")
        else:
            epochs_sem_melhora += 1
            if epochs_sem_melhora >= CLF_PATIENCE:
                print(f"[INFO] Early stopping: sem melhora ha {CLF_PATIENCE} epocas.")
                break

    print(f"\n[INFO] Treino concluido! Melhor val_acc: {best_val_acc:.4f}")
    print(f"[INFO] Pesos e log em: {run_dir}")


if __name__ == "__main__":
    main()
