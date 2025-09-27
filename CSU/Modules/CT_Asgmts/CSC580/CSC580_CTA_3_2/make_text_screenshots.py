#!/usr/bin/env python3
"""
Render text files as PNG "screenshots" using matplotlib, saved into outputs/.
"""
import pathlib
import textwrap
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 10,
})

BASE = pathlib.Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    (OUT / "dataset_tail.txt", OUT / "dataset_tail.png", "Dataset Tail"),
    (OUT / "train_stats_tail.txt", OUT / "train_stats_tail.png", "Train Stats Tail"),
    (OUT / "model_summary_mse.txt", OUT / "model_summary_mse.png", "Model Summary (MSE)") ,
    (OUT / "model_summary_mae.txt", OUT / "model_summary_mae.png", "Model Summary (MAE)"),
    (OUT / "history_tail_mse.txt", OUT / "history_tail_mse.png", "History Tail (MSE-loss)"),
    (OUT / "history_tail_mae.txt", OUT / "history_tail_mae.png", "History Tail (MAE-loss)"),
]


def render_text_image(txt_path: pathlib.Path, png_path: pathlib.Path, title: str):
    if not txt_path.exists():
        print(f"[WARN] Missing {txt_path}")
        return
    text = txt_path.read_text(encoding="utf-8")
    # Wrap long lines for nicer display
    wrapped_lines = []
    for line in text.splitlines():
        if len(line) > 120:
            wrapped_lines.extend(textwrap.wrap(line, width=120))
        else:
            wrapped_lines.append(line)
    content = "\n".join(wrapped_lines)

    # Estimate figure height based on number of lines
    lines = content.count("\n") + 1
    height = max(2, min(30, int(lines * 0.35) + 2))

    fig = plt.figure(figsize=(12, height), dpi=150)
    plt.axis("off")
    plt.title(title, fontsize=12, loc="left")
    plt.text(0.01, 0.98, content, va="top", ha="left", family="monospace")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")


def main():
    for txt, png, title in FILES:
        render_text_image(txt, png, title)

if __name__ == "__main__":
    main()
