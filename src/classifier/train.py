"""
train.py
========
Loop de treino do classificador de kanji (ResNet-18 sobre crops sintéticos).

Roda localmente (CPU) ou no Kaggle (GPU). Local é viável só para testes rápidos
com poucas épocas/classes — o treino real (30 épocas, 1232 classes) deve
rodar em GPU (Kaggle T4), senão o tempo por época fica proibitivo.

Seleção do best.pt: usa o val real de N1 (src/helper/manga109_align_n1.py,
via src.classifier.eval.avaliar_real_n1) quando o conjunto existir em
MANGA109_ALIGN_N1_DIR, não o val_acc sintético -- ver EXPERIMENTS.md
"Validação real de N1" pro porquê (val sintético já mascarou uma regressão
de recall real por medir só a distribuição do próprio gerador). O val
sintético continua decidindo o *early stopping* (patience), que precisa de
um sinal estável época a época; o real_n1 cobre só uma fração das classes e
seria ruidoso demais pra essa decisão. Sem o conjunto real gerado, cai de
volta pro comportamento antigo (best.pt pelo val_acc sintético).

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
    CLF_LABEL_SMOOTHING, CLF_STEM_LEVE,
)
from src.classifier.dataset import build_datasets, build_dataloaders
from src.classifier.model import build_model, count_parameters
from src.classifier.eval import carregar_real_n1, avaliar_real_n1


IS_KAGGLE = (
    os.path.exists("/kaggle/input") or
    "KAGGLE_DATA_PROXY_TOKEN" in os.environ or
    "KAGGLE_KERNEL_RUN_TYPE" in os.environ
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Val real de N1 (src/helper/manga109_align_n1.py) -- usado só pra ESCOLHER
    # o best.pt (ver EXPERIMENTS.md "Validação real de N1"). O val sintético
    # continua sendo quem decide o early stopping (patience) -- o real tem só
    # uma fração das 1232 classes e poucas amostras por classe, ruidoso demais
    # pra decidir "convergiu ou não" época a época, mas é o sinal certo pra
    # decidir QUAL época salvar como checkpoint final.
    real_n1_dados = carregar_real_n1(train_ds.classes)
    if real_n1_dados is None or not real_n1_dados[0]:
        print("[AVISO] Conjunto real de N1 não encontrado -- rode "
              "'python -m src.helper.manga109_align_n1' antes do treino pra "
              "escolher o best.pt pelo sinal real. Caindo de volta pro val_acc "
              "sintético (comportamento antigo).")
        real_n1_dados = None
    else:
        print(f"[INFO] Val real de N1: {len(real_n1_dados[0])} amostras carregadas")

    print(f"[INFO] Construindo modelo (stem_leve={CLF_STEM_LEVE}, "
          f"label_smoothing={CLF_LABEL_SMOOTHING})...")
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

    criterion = nn.CrossEntropyLoss(label_smoothing=CLF_LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=CLF_WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=CLF_EPOCHS)

    run_dir = os.path.join(ROOT_DIR, CLF_PROJECT_NAME, "run")
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best.pt")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "results.csv")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,real_n1_acc,lr\n")

    best_val_acc = 0.0          # decide early stopping (patience) -- val sintético
    best_selecao_acc = 0.0      # decide o best.pt -- real_n1 quando disponível, senão val_acc
    epochs_sem_melhora = 0

    print(f"[INFO] Treinando por ate {CLF_EPOCHS} epocas (patience={CLF_PATIENCE})")
    for epoch in range(1, CLF_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step()

        real_n1_result = None
        if real_n1_dados is not None:
            real_n1_result = avaliar_real_n1(model, train_ds.classes, DEVICE, dados=real_n1_dados)
        metrica_selecao = real_n1_result["top1"] if real_n1_result else val_acc

        dt = time.time() - t0
        lr_atual = scheduler.get_last_lr()[0]
        real_n1_str = f"real_n1_acc={real_n1_result['top1']:.4f} | " if real_n1_result else ""
        print(
            f"[{epoch:03d}/{CLF_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{real_n1_str}"
            f"lr={lr_atual:.2e} | {dt:.1f}s"
        )

        real_n1_csv = f"{real_n1_result['top1']:.6f}" if real_n1_result else ""
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},"
                    f"{real_n1_csv},{lr_atual:.8f}\n")

        checkpoint = {
            "model_state": model.state_dict(),
            "classes": train_ds.classes,
            "epoch": epoch,
            "val_acc": val_acc,
            "real_n1_acc": real_n1_result["top1"] if real_n1_result else None,
            # stride/Identity do stem_leve nao aparecem no state_dict (nao sao
            # peso, so config de forward) -- sem guardar isso aqui, carregar o
            # checkpoint em outra maquina/sessao (onde CLF_STEM_LEVE local
            # default e' False) reconstroi a arquitetura ERRADA e os pesos da
            # conv1 carregam sem erro de shape, mas a rede fica incoerente
            # (bug real encontrado ao validar: recall caiu pra 0%).
            "stem_leve": CLF_STEM_LEVE,
        }
        torch.save(checkpoint, last_path)

        # Patience: baseado no val sintético (estável, roda em 100% das classes
        # -- o real_n1 só cobre uma fração, ruidoso demais pra decidir "parou
        # de convergir" época a época).
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_sem_melhora = 0
        else:
            epochs_sem_melhora += 1

        # Seleção do best.pt: usa o real_n1 quando disponível (ver
        # EXPERIMENTS.md "Validação real de N1" -- é o sinal que reflete o
        # mundo real, o val_acc sintético já mascarou uma regressão de
        # verdade antes por só medir a distribuição do próprio gerador).
        if metrica_selecao > best_selecao_acc:
            best_selecao_acc = metrica_selecao
            torch.save(checkpoint, best_path)
            fonte = "real_n1_acc" if real_n1_result else "val_acc (fallback, sem real_n1)"
            print(f"          -> novo melhor {fonte}: {metrica_selecao:.4f} (salvo em {best_path})")

        if epochs_sem_melhora >= CLF_PATIENCE:
            print(f"[INFO] Early stopping: sem melhora no val sintético ha {CLF_PATIENCE} epocas.")
            break

    print(f"\n[INFO] Treino concluido! Melhor val_acc: {best_val_acc:.4f} | "
          f"Melhor metrica de selecao do best.pt: {best_selecao_acc:.4f}")
    print(f"[INFO] Pesos e log em: {run_dir}")


if __name__ == "__main__":
    main()
