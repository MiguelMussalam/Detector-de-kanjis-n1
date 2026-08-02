"""
eval.py
=======
Avaliação sistemática do classificador em múltiplos val sets:
  1. Val sintético held-out (o mesmo usado durante o treino)
  2. ETL9B filtrado por N1 (proxy fora do domínio — manuscrito real)
  3. Val real de N1 (Manga109 alinhado + filtrado por manga-ocr — ver
     src/helper/manga109_align_n1.py). Esse é o sinal que mais importa pra
     decidir qual checkpoint promover -- os outros dois são proxies.

Uso:
    python -m src.classifier.eval                  # avalia em tudo
    python -m src.classifier.eval --only synth      # só sintético
    python -m src.classifier.eval --only etl9       # só ETL9
    python -m src.classifier.eval --only real_n1    # só val real de N1
"""

import argparse
import csv
import os
import random
from collections import Counter

import torch
from torch.utils.data import DataLoader
from PIL import Image

from config import (
    CLASSIFIER_INPUT_SIZE,
    CLF_WEIGHTS_PATH,
    CLF_BATCH_SIZE, CLF_NUM_WORKERS,
    CLF_OUTROS_LABEL,
    ROOT_DIR,
    VISUAL_COMPARACAO_DIR,
)
from src.classifier.model import build_model
from src.classifier.dataset import build_datasets, build_transform, index_to_kanji

CLF_EVAL_DIR = os.path.join(ROOT_DIR, "data", "classifier_eval")

def load_classifier(weights_path: str, device: str):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pesos do classificador não encontrados: {weights_path}\n"
            f"Treine primeiro (notebooks/02_classifier_train.ipynb) ou ajuste "
            f"KD_CLF_WEIGHTS_PATH."
        )
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    # stem_leve muda stride/Identity (nao pesa no state_dict) -- precisa vir
    # do proprio checkpoint, senao a arquitetura reconstruida pode nao bater
    # com a que treinou os pesos (mesmo bug ja corrigido em
    # src/pipeline/inference.py -- carregar sem isso deu recall 0% na validacao).
    stem_leve = ckpt.get("stem_leve", False)
    model = build_model(num_classes=len(classes), stem_leve=stem_leve)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes

@torch.no_grad()
def eval_synthetic(model, classes, device):
    print("\n=== Val sintético held-out ===")
    _, val_ds = build_datasets()

    # Guarda de sanidade: os indices de classe do ImageFolder recem-construido
    # so tem o mesmo significado que os aprendidos pelo modelo se as classes
    # baterem exatamente (mesmo conjunto, mesma ordem). Se data/classifier/val
    # foi regenerado parcialmente (ex: com --limit) depois do treino, os indices
    # nao alinham e a acuracia calculada seria silenciosamente sem sentido.
    if val_ds.classes != classes:
        raise RuntimeError(
            f"Classes do val_ds ({len(val_ds.classes)}) nao batem com as classes "
            f"salvas no checkpoint ({len(classes)}). O diretorio de dados provavelmente "
            f"foi regenerado parcialmente (ex: generate_crops --limit) depois do treino. "
            f"Regenere o dataset completo antes de avaliar."
        )

    loader = DataLoader(
        val_ds, batch_size=CLF_BATCH_SIZE, shuffle=False,
        num_workers=CLF_NUM_WORKERS, pin_memory=True,
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        top5 = logits.topk(5, dim=1).indices
        correct_top1 += (top5[:, 0] == labels).sum().item()
        correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    acc1 = correct_top1 / total
    acc5 = correct_top5 / total
    print(f"Amostras: {total}")
    print(f"Top-1 accuracy: {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"Top-5 accuracy: {acc5:.4f} ({acc5*100:.2f}%)")
    return {"top1": acc1, "top5": acc5, "n": total}

@torch.no_grad()
def eval_etl9(model, classes, device):
    print("\n=== ETL9B filtrado por N1 ===")
    from src.helper.etl9 import load_etl9

    # Filtra ETL9 apenas para os kanji que o classificador conhece (ignora a
    # classe CLF_OUTROS_LABEL, que não é um kanji e não existe no ETL9)
    kanji_classes = [c for c in classes if c != CLF_OUTROS_LABEL]
    target_kanjis = {chr(int(c.replace("U+", ""), 16)) for c in kanji_classes}
    images, labels, stats = load_etl9(target_kanjis=target_kanjis,
                                       resize=(CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE))

    print(f"Cobertura ETL9->N1: {stats['n_classes_encontradas']}/{stats['n_classes_target']} classes")
    print(f"Amostras utilizáveis: {stats['n_samples']}")

    kanji_to_idx = {}
    for i, c in enumerate(classes):
        if c == CLF_OUTROS_LABEL:
            continue
        kanji = chr(int(c.replace("U+", ""), 16))
        kanji_to_idx[kanji] = i

    transform = build_transform()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    errors_by_class = Counter()

    for i in range(0, len(images), CLF_BATCH_SIZE):
        batch_imgs = images[i:i + CLF_BATCH_SIZE]
        batch_labels = labels[i:i + CLF_BATCH_SIZE]

        # images ja vem em uint8 (H, W) — ver src/helper/etl9.py, _para_uint8
        tensors = [transform(Image.fromarray(img_np).convert("L")) for img_np in batch_imgs]
        x = torch.stack(tensors).to(device)

        target_idx = torch.tensor(
            [kanji_to_idx[lbl] for lbl in batch_labels],
            dtype=torch.long, device=device
        )

        logits = model(x)
        top5 = logits.topk(5, dim=1).indices

        correct_mask = (top5[:, 0] == target_idx)
        correct_top1 += correct_mask.sum().item()
        correct_top5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()
        total += len(batch_labels)

        for lbl, ok in zip(batch_labels, correct_mask.cpu().tolist()):
            if not ok:
                errors_by_class[lbl] += 1

    acc1 = correct_top1 / total
    acc5 = correct_top5 / total
    print(f"Top-1 accuracy: {acc1:.4f} ({acc1*100:.2f}%)")
    print(f"Top-5 accuracy: {acc5:.4f} ({acc5*100:.2f}%)")

    print("\nTop 10 kanji com mais erros:")
    for kanji, n_err in errors_by_class.most_common(10):
        print(f"  {kanji} ({ord(kanji):04X}): {n_err} erros")

    return {"top1": acc1, "top5": acc5, "n": total, "cobertura": stats}


def carregar_real_n1(classes):
    """
    Lista (path, target_idx) do conjunto real de N1 (ver manga109_align_n1.py),
    já mapeado pros índices de classe do checkpoint. Retorna None se o
    diretório não existe (ainda não gerado). Separado de `avaliar_real_n1`
    pra quem chama a cada época (train.py) poder listar os arquivos uma vez
    só, não a cada chamada.
    """
    from glob import glob
    from config import MANGA109_ALIGN_N1_DIR

    if not os.path.isdir(MANGA109_ALIGN_N1_DIR):
        return None

    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, target_idxs = [], []
    for codepoint_dir in sorted(os.listdir(MANGA109_ALIGN_N1_DIR)):
        idx = class_to_idx.get(codepoint_dir)
        if idx is None:
            continue  # classe fora do que esse checkpoint conhece (raro)
        for p in glob(os.path.join(MANGA109_ALIGN_N1_DIR, codepoint_dir, "*.png")):
            paths.append(p)
            target_idxs.append(idx)
    return paths, target_idxs


@torch.no_grad()
def avaliar_real_n1(model, classes, device, dados=None):
    """
    Núcleo de avaliação no conjunto real de N1 -- sem print, retorna None se
    o conjunto não existe. `dados` (saída de `carregar_real_n1`) pode ser
    passado pronto pra evitar relistar o diretório a cada chamada (usado por
    train.py, uma vez por época). Se omitido, carrega na hora (uso avulso,
    ver `eval_real_n1`).
    """
    if dados is None:
        dados = carregar_real_n1(classes)
    if dados is None:
        return None
    paths, target_idxs = dados
    if not paths:
        return None

    transform = build_transform()
    correct_top1 = 0
    correct_top5 = 0
    errors_by_class = Counter()

    for i in range(0, len(paths), CLF_BATCH_SIZE):
        batch_paths = paths[i:i + CLF_BATCH_SIZE]
        batch_targets = target_idxs[i:i + CLF_BATCH_SIZE]

        tensors = [transform(Image.open(p).convert("L")) for p in batch_paths]
        x = torch.stack(tensors).to(device)
        target_idx = torch.tensor(batch_targets, dtype=torch.long, device=device)

        logits = model(x)
        top5 = logits.topk(5, dim=1).indices

        correct_mask = (top5[:, 0] == target_idx)
        correct_top1 += correct_mask.sum().item()
        correct_top5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()

        for idx, ok in zip(batch_targets, correct_mask.cpu().tolist()):
            if not ok:
                errors_by_class[classes[idx]] += 1

    total = len(paths)
    return {
        "top1": correct_top1 / total,
        "top5": correct_top5 / total,
        "n": total,
        "n_classes": len(set(target_idxs)),
        "errors_by_class": errors_by_class,
    }


def eval_real_n1(model, classes, device):
    """
    Val real de N1 (src/helper/manga109_align_n1.py) -- crops reais do
    Manga109, alinhados por heurística e filtrados por concordância com o
    manga-ocr (ver docstring do script gerador e EXPERIMENTS.md "Validação
    real de N1"). Ao contrário do val sintético, isso mede o que realmente
    importa: acerto em kanji de mangá de verdade, não na distribuição do
    próprio gerador -- é o sinal que deveria decidir qual checkpoint promover,
    não só o val_acc sintético (ver EXPERIMENTS.md "Regressão do checkpoint
    2026-08-01" para um caso real onde confiar só no val_acc sintético
    escondeu uma regressão). Wrapper com relatório em cima de `avaliar_real_n1`.
    """
    print("\n=== Val real de N1 (Manga109 alinhado + manga-ocr) ===")
    from config import MANGA109_ALIGN_N1_DIR

    r = avaliar_real_n1(model, classes, device)
    if r is None:
        print(f"[AVISO] {MANGA109_ALIGN_N1_DIR} não existe (ou está vazio) -- rode "
              f"'python -m src.helper.manga109_align_n1' primeiro. Pulando.")
        return None

    print(f"Amostras: {r['n']} | Classes cobertas: {r['n_classes']}")
    print(f"Top-1 accuracy: {r['top1']:.4f} ({r['top1']*100:.2f}%)")
    print(f"Top-5 accuracy: {r['top5']:.4f} ({r['top5']*100:.2f}%)")

    print("\nTop 10 classes com mais erros:")
    for codepoint, n_err in r["errors_by_class"].most_common(10):
        print(f"  {codepoint}: {n_err} erros")

    return {"top1": r["top1"], "top5": r["top5"], "n": r["n"]}


@torch.no_grad()
def eval_confusoes(model, classes, device, out_dir: str = CLF_EVAL_DIR,
                   n_exemplos: int = 30, seed: int = 42):
    """
    Análise qualitativa de erro no val sintético (dado limpo, rótulo 1-pra-1
    -- ao contrário de página real, onde só sabemos multiset por linha, não
    qual caixa era qual). Responde: os erros do classificador são confusões
    entre kanji visualmente parecidos (esperado, até saudável) ou essencialmente
    aleatórios (sinal de problema mais sério)? Ver objetivo específico 5 da
    proposta da IC.

    Salva:
      - {out_dir}/confusoes.csv: pra cada kanji com erro, top-3 classes mais
        confundidas + contagem + % dos erros daquela classe.
      - {out_dir}/acuracia_por_classe.csv: acurácia individual de TODAS as
        1232 classes N1 (não só as que erraram) -- usado por
        `src/pesquisa/frequencia_vs_acuracia.py` pra cruzar com a frequência
        real de cada kanji no corpus Manga109.
      - {VISUAL_COMPARACAO_DIR}/nosso_pipeline.png: grade de ~n_exemplos crops
        errados, imagem original (não normalizada) + rótulo verdadeiro/previsto.
    """
    print("\n=== Análise de confusão (val sintético) ===")
    # So precisa do val (nao do train) -- monta direto em vez de build_datasets(),
    # que exige train/val com o mesmo conjunto de classes (checagem pensada pro
    # treino de verdade, nao se aplica aqui: o split de treino local pode estar
    # em qualquer estado, ex: um subconjunto de teste, sem afetar essa analise).
    from config import CLASSIFIER_VAL_DIR
    from torchvision.datasets import ImageFolder
    val_ds = ImageFolder(root=CLASSIFIER_VAL_DIR, transform=build_transform())

    if val_ds.classes != classes:
        raise RuntimeError(
            f"Classes do val_ds ({len(val_ds.classes)}) nao batem com as classes "
            f"salvas no checkpoint ({len(classes)}). Ver mesma checagem em eval_synthetic."
        )

    loader = DataLoader(
        val_ds, batch_size=CLF_BATCH_SIZE, shuffle=False,
        num_workers=CLF_NUM_WORKERS, pin_memory=True,
    )

    erros = []  # (indice_global, true_idx, pred_idx)
    total_por_classe = Counter()
    corretos_por_classe = Counter()
    idx_global = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu()
        for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
            total_por_classe[true_idx] += 1
            if pred_idx == true_idx:
                corretos_por_classe[true_idx] += 1
            else:
                erros.append((idx_global, true_idx, pred_idx))
            idx_global += 1
            total += 1

    n_erros = len(erros)
    print(f"Amostras: {total} | Erros: {n_erros} ({100*n_erros/max(1,total):.2f}%)")

    os.makedirs(out_dir, exist_ok=True)

    # Tabela de confusao: por classe verdadeira, top-3 classes mais confundidas
    contagem_pares = Counter((t, p) for _, t, p in erros)
    contagem_por_classe = Counter(t for _, t, _ in erros)

    linhas_csv = []
    for true_idx, total_erros_classe in contagem_por_classe.most_common():
        pares_dessa_classe = sorted(
            ((p, c) for (t, p), c in contagem_pares.items() if t == true_idx),
            key=lambda x: -x[1],
        )[:3]
        for pred_idx, contagem in pares_dessa_classe:
            linhas_csv.append({
                "kanji_verdadeiro": index_to_kanji(classes, true_idx),
                "classe_verdadeira": classes[true_idx],
                "kanji_confundido_com": index_to_kanji(classes, pred_idx),
                "classe_confundida": classes[pred_idx],
                "contagem": contagem,
                "pct_dos_erros_da_classe": round(100 * contagem / total_erros_classe, 1),
            })

    csv_path = os.path.join(out_dir, "confusoes.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas_csv[0].keys()) if linhas_csv else [
            "kanji_verdadeiro", "classe_verdadeira", "kanji_confundido_com",
            "classe_confundida", "contagem", "pct_dos_erros_da_classe",
        ])
        writer.writeheader()
        writer.writerows(linhas_csv)
    print(f"[INFO] Tabela de confusao salva em: {csv_path}")

    # Acuracia individual de TODAS as classes N1 (nao so quem errou) -- pra
    # cruzar depois com a frequencia real no corpus (kanji raro no corpus
    # tem acuracia pior, ou o sintetico compensa igual pra todo mundo?).
    acuracia_path = os.path.join(out_dir, "acuracia_por_classe.csv")
    with open(acuracia_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kanji", "classe", "total_val", "corretos", "acuracia_pct"])
        writer.writeheader()
        for idx, classe in enumerate(classes):
            if classe == CLF_OUTROS_LABEL:
                continue
            n_total = total_por_classe.get(idx, 0)
            if n_total == 0:
                continue
            n_corretos = corretos_por_classe.get(idx, 0)
            writer.writerow({
                "kanji": index_to_kanji(classes, idx),
                "classe": classe,
                "total_val": n_total,
                "corretos": n_corretos,
                "acuracia_pct": round(100 * n_corretos / n_total, 1),
            })
    print(f"[INFO] Acuracia por classe salva em: {acuracia_path}")

    # Grade visual de exemplos errados (imagem original, nao normalizada).
    # Pre-amostra os ERROS (tuplas leves) antes de abrir imagem nenhuma -- evita
    # Image.open() em milhares de crops que a amostragem ia descartar de qualquer jeito.
    png_path = None
    if erros:
        from src.helper.grade_visual import salvar_grade_visual

        rng = random.Random(seed)
        amostra = rng.sample(erros, min(n_exemplos, len(erros)))
        itens = [{
            "imagem": Image.open(val_ds.samples[idx_g][0]).convert("L"),
            "titulo": (f"real: {index_to_kanji(classes, true_idx)} ({classes[true_idx]})\n"
                       f"previsto: {index_to_kanji(classes, pred_idx)} ({classes[pred_idx]})"),
            "cor": "red",
            "cmap": "gray",
        } for idx_g, true_idx, pred_idx in amostra]

        png_path = os.path.join(VISUAL_COMPARACAO_DIR, "nosso_pipeline.png")
        salvar_grade_visual(itens, png_path, n=len(itens), seed=seed, cols=6, figsize_cel=(2.2, 2.6))
    else:
        print("[INFO] Nenhum erro no val sintetico -- nada pra visualizar.")

    return {"n_erros": n_erros, "n_total": total, "csv": csv_path, "png": png_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["synth", "etl9", "real_n1", "confusao"], default=None,
                        help="Avaliar apenas um dos conjuntos. 'confusao' so roda se "
                             "explicitamente pedido (nao faz parte do 'avaliar tudo' default).")
    parser.add_argument("--weights", type=str, default=CLF_WEIGHTS_PATH,
                        help="Caminho dos pesos do classificador.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Carregando classificador de {args.weights}")

    model, classes = load_classifier(args.weights, device)
    print(f"[INFO] Modelo carregado: {len(classes)} classes")

    if args.only == "confusao":
        eval_confusoes(model, classes, device)
        return

    results = {}
    if args.only in (None, "synth"):
        results["synth"] = eval_synthetic(model, classes, device)

    if args.only in (None, "etl9"):
        results["etl9"] = eval_etl9(model, classes, device)

    if args.only in (None, "real_n1"):
        r = eval_real_n1(model, classes, device)
        if r is not None:
            results["real_n1"] = r

    if len(results) > 1:
        print("\n" + "=" * 50)
        print("Resumo comparativo")
        print("=" * 50)
        for name, r in results.items():
            print(f"  {name:8s}: top-1 = {r['top1']*100:.2f}%  |  n = {r['n']}")

        if "etl9" in results:
            gap = results["synth"]["top1"] - results["etl9"]["top1"]
            print(f"\nGap sintético -> ETL9: {gap*100:.2f} pontos")
            if gap < 0.15:
                print("  -> Modelo generaliza bem entre domínios.")
            elif gap < 0.30:
                print("  -> Gap moderado; sugere aumentar variedade de fontes ou domain randomization.")
            else:
                print("  -> Gap severo; modelo aprendeu features específicas de fonte, não de kanji.")

        if "real_n1" in results:
            gap_real = results["synth"]["top1"] - results["real_n1"]["top1"]
            print(f"\nGap sintético -> real N1 (Manga109): {gap_real*100:.2f} pontos")
            print("  -> Esse é o sinal que deveria decidir qual checkpoint promover, "
                  "não o val_acc sintético sozinho (ver EXPERIMENTS.md).")


if __name__ == "__main__":
    main()
