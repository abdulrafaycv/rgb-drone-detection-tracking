# Training Process

## Stage 1 — Base Training

The initial YOLO detector was trained using the custom RGB drone dataset.

### Result

* mAP50: **97.0%**
* mAP50-95: **64.6%**

The training configuration used for the experiment is available under:

```text
training/base/
```

---

## Stage 2 — Fine-Tuning

Additional data containing more challenging visual conditions was collected.

One of the important conditions targeted during this stage was severe motion blur.

The detector was then fine-tuned using the expanded dataset.

### Result

* mAP50: **94.5%**
* mAP50-95: **62.2%**

The fine-tuned model and training configuration are available under:

```text
training/finetuned/
```

## Evaluation

Training and evaluation plots are stored with each corresponding training run.

These include the available:

* Training curves
* Precision/Recall curves
* F1 curves
* Confusion matrices
* Precision curves
* Recall curves
* PR curves

## Important Note

The base and fine-tuned results should be interpreted according to their
respective datasets and validation conditions. A direct comparison of the
numbers alone does not necessarily indicate a reduction in real-world
performance because the fine-tuning stage introduced more difficult imagery.
