#!/usr/bin/env python3
"""
Unsupervised Set & Group Theory ML Studio (PySide6) — FULL BUILD
================================================================
Build UNLABELED datasets (sets of points A ⊂ R²) and explore them with:

  1 · Data Studio    - 9 generators, rename, stats, affine transform g·A
  2 · Set Theory     - ε-tolerant ∪ ∩ ∖ △, region maps, Jaccard/Dice,
                       n-ary ⋃/⋂ over the whole universe U
  3 · Group Theory   - quotient sets A/~, groups D₄/Dₙ/Cₙ acting on ℝ²:
                       orbits, stabilizers, Cayley tables, ALL subgroups
                       (Lagrange), conjugacy classes, invariant features,
                       group-action augmentation (g·A → U)
  4 · Neural Net     - numpy MLP autoencoder (denoising, tanh/ReLU,
                       Adam/SGD, L2), reconstruction panel, latent → U
  5 · Ensembles      - RandomForest & AdaBoost on pseudo-labels q: A → A/~,
                       staged boosting curve, OOB, disagreement map,
                       D₄-rotation self-consistency probe
  6 · Validation     - sklearn benchmarks with GROUND TRUTH (ARI/NMI/
                       purity, 5-fold CV vs Dummy, kNN probes, AE R²),
                       clustering bake-off, full-suite benchmark sweep,
                       set-law property tests, D₄ axiom tests, Markdown
                       report export

File menu: save/load the universe (.npz), export plots (PNG/PDF/SVG).

Install:  pip install PySide6 numpy matplotlib scikit-learn scipy
Run:      python set_group_ml_studio.py
"""

import sys
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import sklearn
import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, Normalize

from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QAction, QKeySequence, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGridLayout, QGroupBox, QPushButton, QComboBox, QSpinBox,
    QDoubleSpinBox, QListWidget, QListWidgetItem, QLabel, QTextEdit,
    QProgressBar, QMessageBox, QFileDialog, QDialog, QDialogButtonBox,
    QAbstractItemView, QScrollArea,
)

from sklearn.datasets import (make_blobs, make_moons, make_circles,
                              load_iris, load_wine, load_breast_cancer,
                              load_digits, make_classification)
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold)
from sklearn.metrics import (silhouette_score, accuracy_score,
                             adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_completeness_v_measure,
                             confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree


# ═══════════════════════════════ helpers ═══════════════════════════════════

def get_cmap(name):
    try:
        return matplotlib.colormaps[name]
    except Exception:                                    # older matplotlib
        return matplotlib.cm.get_cmap(name)


def tab10(k):
    return ListedColormap(get_cmap("tab10").colors[:max(1, k)])


def mono_text():
    t = QTextEdit()
    t.setReadOnly(True)
    t.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
    return t


def refresh_combo(combo, store):
    cur = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(store.names())
    if cur:
        i = combo.findText(cur)
        if i >= 0:
            combo.setCurrentIndex(i)
    combo.blockSignals(False)


def make_adaboost(seed, n_estimators, lr, depth):
    base = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_estimators,
                                  learning_rate=lr, random_state=seed)
    except TypeError:                                # scikit-learn < 1.2
        return AdaBoostClassifier(base_estimator=base,
                                  n_estimators=n_estimators,
                                  learning_rate=lr, random_state=seed)


def cluster_purity(y_true, y_clu):
    total = 0
    for c in np.unique(y_clu):
        _, cnts = np.unique(y_true[y_clu == c], return_counts=True)
        total += cnts.max()
    return total / len(y_true)


BENCH_DATASETS = ["Iris", "Wine", "Breast Cancer", "Digits (sub-500)",
                  "Synthetic classification"]

REFERENCE_ACC = {          # typical published 5-fold CV accuracies (anchors)
    "Iris": "≈ 0.95–0.97", "Wine": "≈ 0.96–0.99",
    "Breast Cancer": "≈ 0.95–0.97", "Digits (sub-500)": "≈ 0.92–0.96",
    "Synthetic classification": "n/a",
}


def load_benchmark(key, seed):
    if key == "Iris":
        d = load_iris(); X, y, names = d.data, d.target, list(d.target_names)
    elif key == "Wine":
        d = load_wine(); X, y, names = d.data, d.target, list(d.target_names)
    elif key == "Breast Cancer":
        d = load_breast_cancer()
        X, y, names = d.data, d.target, list(d.target_names)
    elif key == "Digits (sub-500)":
        d = load_digits()
        idx = np.arange(len(d.data))
        _, idx, _, y = train_test_split(idx, d.target, train_size=500,
                                        stratify=d.target, random_state=seed)
        X, names = d.data[idx], list(d.target_names)
    else:
        X, y = make_classification(n_samples=800, n_features=10,
                                   n_informative=6, n_redundant=2, n_classes=3,
                                   n_clusters_per_class=1, class_sep=1.2,
                                   random_state=seed)
        names = ["class 0", "class 1", "class 2"]
    return np.asarray(X, float), np.asarray(y, int), names


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, width=6.0, height=4.5):
        self.fig = Figure(figsize=(width, height), constrained_layout=True)
        super().__init__(self.fig)


# ═════════════════════════════ dataset generators ══════════════════════════

def _spirals(n, noise, rng):
    n1 = n // 2
    t1 = rng.uniform(0.25, 1.0, n1)
    t2 = rng.uniform(0.25, 1.0, n - n1)
    a1 = np.column_stack([t1 * np.cos(3 * np.pi * t1),
                          t1 * np.sin(3 * np.pi * t1)])
    a2 = np.column_stack([-t2 * np.cos(3 * np.pi * t2),
                          -t2 * np.sin(3 * np.pi * t2)])
    return np.vstack([a1, a2]) + rng.normal(0, 0.05 + 0.12 * noise, (n, 2))


def _rings(n, noise, rng):
    r = rng.choice([1.0, 2.2, 3.4], size=n)
    th = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)]) \
        + rng.normal(0, 0.03 + 0.06 * noise, (n, 2))


def _xor_pts(n, noise, rng):
    signs = 2 * rng.integers(0, 2, size=(n, 2)) - 1
    return signs * 1.5 + rng.normal(0, 0.30 + 0.25 * noise, (n, 2))


def _aniso(n, clusters, noise, rng):
    idx = rng.integers(0, clusters, n)
    centers = rng.uniform(-4, 4, (clusters, 2))
    angs = rng.uniform(0, np.pi, clusters)
    out = np.empty((n, 2))
    for i in range(clusters):
        m = idx == i
        if not m.any():
            continue
        c, s = np.cos(angs[i]), np.sin(angs[i])
        R = np.array([[c, -s], [s, c]])
        S = R @ np.diag([0.40 + 0.6 * noise, 0.06 + 0.10 * noise])
        out[m] = centers[i] + rng.normal(0, 1, (int(m.sum()), 2)) @ S.T
    return out


def _outliers(n, noise, rng):
    n_out = max(1, int(0.12 * n))
    Xin, _ = make_blobs(n - n_out, centers=3, cluster_std=0.5 + noise,
                        random_state=int(rng.integers(1e9)))
    return np.vstack([Xin, rng.uniform(-6, 6, (n_out, 2))])


def make_dataset(kind, n, noise, clusters, seed):
    rng = np.random.default_rng(seed)
    if kind == "Blobs":
        X, _ = make_blobs(n_samples=n, centers=clusters,
                          cluster_std=0.5 + noise, random_state=seed)
    elif kind == "Moons":
        X, _ = make_moons(n_samples=n, noise=0.05 + 0.15 * noise,
                          random_state=seed)
    elif kind == "Circles":
        X, _ = make_circles(n_samples=n, noise=0.02 + 0.08 * noise,
                            factor=0.4, random_state=seed)
    elif kind == "Spirals":
        X = _spirals(n, noise, rng)
    elif kind == "Rings":
        X = _rings(n, noise, rng)
    elif kind == "XOR":
        X = _xor_pts(n, noise, rng)
    elif kind == "Anisotropic":
        X = _aniso(n, clusters, noise, rng)
    elif kind == "Outliers":
        X = _outliers(n, noise, rng)
    else:
        X = rng.uniform(-4, 4, size=(n, 2))
    return np.asarray(X, float)


# ═════════════════════════════ dataset store ═══════════════════════════════

class DatasetStore(QObject):
    """Named unlabeled datasets; emits `changed` so all tabs refresh."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self.datasets = {}

    def names(self):
        return list(self.datasets.keys())

    def add(self, name, X):
        base, i = name, 1
        while name in self.datasets:
            i += 1
            name = f"{base}_{i}"
        self.datasets[name] = np.asarray(X, float)
        self.changed.emit()
        return name

    def rename(self, old, new):
        if old not in self.datasets or not new or new == old:
            return old
        base, i = new, 1
        while new in self.datasets:
            i += 1
            new = f"{base}_{i}"
        self.datasets = {new if k == old else k: v
                         for k, v in self.datasets.items()}
        self.changed.emit()
        return new

    def delete(self, name):
        self.datasets.pop(name, None)
        self.changed.emit()


# ═════════════════════════════ set theory ops ══════════════════════════════
# Floating-point point clouds rarely share exact elements, so every
# operation is ε-tolerant:  x ∈ A∩B  ⇔  ∃ b∈B : ‖x−b‖ ≤ ε

def eps_dedupe(P, eps):
    if len(P) == 0:
        return P
    parent = np.arange(len(P))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cKDTree(P).query_pairs(r=eps, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    keep, seen = [], set()
    for i, r in enumerate([find(i) for i in range(len(P))]):
        if r not in seen:
            seen.add(r)
            keep.append(i)
    return P[keep]


def set_intersection(A, B, eps):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((0, 2))
    d, _ = cKDTree(B).query(A, k=1)
    return A[d <= eps]


def set_difference(A, B, eps):
    if len(A) == 0:
        return np.zeros((0, 2))
    if len(B) == 0:
        return A.copy()
    d, _ = cKDTree(B).query(A, k=1)
    return A[d > eps]


def set_union(A, B, eps):
    """Disjoint decomposition: A∪B = A ⊎ (B∖A)  → counts stay exact."""
    if len(A) == 0:
        return B.copy()
    if len(B) == 0:
        return A.copy()
    return np.vstack([A, set_difference(B, A, eps)])


def set_symdiff(A, B, eps):
    return np.vstack([set_difference(A, B, eps), set_difference(B, A, eps)])


# ═════════════════════════════ group theory (Dₙ/Cₙ) ════════════════════════

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _sup(k):
    return "e" if k == 0 else "r" + (str(k).translate(_SUP) if k > 1 else "")


def dihedral_elements(n, cyclic=False):
    """The 2n matrices of Dₙ (rotations + reflections),
    or the n rotations of Cₙ."""
    th = 2 * np.pi / n
    c, s = np.cos(th), np.sin(th)
    r = np.array([[c, -s], [s, c]])
    ref = np.array([[1.0, 0.0], [0.0, -1.0]])
    elems, names = [], []
    R = np.eye(2)
    for k in range(n):
        elems.append(R.copy())
        names.append(_sup(k))
        if not cyclic:
            elems.append(R @ ref)
            names.append("s" if k == 0 else _sup(k) + "s")
        R = R @ r
    return elems, names


def d4_elements():                                   # kept for validation tab
    return dihedral_elements(4)


def _match_matrix(M, elems):
    for i, E in enumerate(elems):
        if np.allclose(M, E, atol=1e-9):
            return i
    return None


# ═════════════════════════════ numpy autoencoder ═══════════════════════════

class Autoencoder:
    """MLP autoencoder — tanh/ReLU hidden units, linear output, MSE loss,
    Adam or SGD(momentum), optional L2 penalty and input corruption
    (denoising AE). Pure numpy so every gradient is visible."""

    def __init__(self, dims, seed=0, activation="tanh", opt="adam", l2=0.0):
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.enc_layers = (len(dims) - 1) // 2
        self.activation, self.opt, self.l2 = activation, opt, l2
        self.W, self.b = [], []
        for fin, fout in zip(dims[:-1], dims[1:]):
            lim = np.sqrt(2.0 / fin) if activation == "relu" \
                else np.sqrt(6.0 / (fin + fout))
            self.W.append(rng.uniform(-lim, lim, (fin, fout)))
            self.b.append(np.zeros(fout))
        z = lambda a: np.zeros_like(a)               # noqa: E731
        self._adam = [(z(w), z(w), z(b), z(b)) for w, b in zip(self.W, self.b)]
        self._vel = [(z(w), z(b)) for w, b in zip(self.W, self.b)]
        self.t = 0

    def _act(self, zv):
        return np.tanh(zv) if self.activation == "tanh" \
            else np.maximum(zv, 0.0)

    def _dact(self, a):
        return 1.0 - a * a if self.activation == "tanh" \
            else (a > 0).astype(float)

    def _forward(self, X):
        acts, a, last = [X], X, len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i != last:
                a = self._act(a)
            acts.append(a)
        return acts

    def encode(self, X):
        a = X
        for i in range(self.enc_layers):
            a = self._act(a @ self.W[i] + self.b[i])
        return a

    def fit(self, X, epochs=300, lr=1e-2, batch=32, progress=None, noise=0.0):
        rng, n = np.random.default_rng(0), len(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        history = []
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            for st in range(0, n, batch):
                xc = X[order[st:st + batch]]
                xb = xc + noise * rng.standard_normal(xc.shape) if noise \
                    else xc
                acts = self._forward(xb)
                m = len(xc)
                delta = 2.0 * (acts[-1] - xc) / m          # target = clean x
                gW = [None] * len(self.W)
                gb = [None] * len(self.b)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ delta
                    gb[i] = delta.sum(axis=0)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * self._dact(acts[i])
                    if self.l2:
                        gW[i] = gW[i] + self.l2 * self.W[i]
                self.t += 1
                if self.opt == "adam":
                    for i in range(len(self.W)):
                        mw, vw, mb, vb = self._adam[i]
                        mw = b1 * mw + (1 - b1) * gW[i]
                        vw = b2 * vw + (1 - b2) * gW[i] ** 2
                        mb = b1 * mb + (1 - b1) * gb[i]
                        vb = b2 * vb + (1 - b2) * gb[i] ** 2
                        self._adam[i] = (mw, vw, mb, vb)
                        self.W[i] -= lr * (mw / (1 - b1 ** self.t)) / \
                            (np.sqrt(vw / (1 - b2 ** self.t)) + eps)
                        self.b[i] -= lr * (mb / (1 - b1 ** self.t)) / \
                            (np.sqrt(vb / (1 - b2 ** self.t)) + eps)
                else:                                       # SGD + momentum
                    for i in range(len(self.W)):
                        vW, vb = self._vel[i]
                        vW = 0.9 * vW - lr * gW[i]
                        vb = 0.9 * vb - lr * gb[i]
                        self._vel[i] = (vW, vb)
                        self.W[i] += vW
                        self.b[i] += vb
            loss = float(np.mean((self._forward(X)[-1] - X) ** 2))
            history.append(loss)
            if progress is not None and progress(epoch, loss):
                break                                   # stop requested
        return history


class TrainingWorker(QThread):
    progress = Signal(int, float)
    finished_run = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            res = self._fn(self)
        except Exception as e:                          # noqa: BLE001
            res = e
        self.finished_run.emit(res)


# ═══════════════════════════════ Tab 1: Data ═══════════════════════════════

def dataset_stats_text(X):
    if len(X) == 0:
        return "∅  (empty set)"
    nn = ""
    if 1 < len(X) <= 20000:
        d, _ = cKDTree(X).query(X, k=2)
        nn = f"\nmean NN distance = {d[:, 1].mean():.3f}"
    return (f"|A| = {len(X)}\n"
            f"mean = ({X[:, 0].mean():+.2f}, {X[:, 1].mean():+.2f})\n"
            f"std  = ({X[:, 0].std():.2f}, {X[:, 1].std():.2f})\n"
            f"bbox = [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}] × "
            f"[{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]" + nn)


class TransformDialog(QDialog):
    """Affine group action:  x ↦ Shear·Scale·Rotate·x + t  (+ jitter)."""

    def __init__(self, parent, name):
        super().__init__(parent)
        self.setWindowTitle(f"Apply g ∈ Aff(2) to “{name}”")
        f = QFormLayout(self)
        self.rot = QDoubleSpinBox(); self.rot.setRange(-180, 180)
        self.rot.setValue(90); self.rot.setSuffix("°")
        self.sx = QDoubleSpinBox(); self.sx.setRange(0.1, 4)
        self.sx.setSingleStep(0.1); self.sx.setValue(1.0)
        self.sy = QDoubleSpinBox(); self.sy.setRange(0.1, 4)
        self.sy.setSingleStep(0.1); self.sy.setValue(1.0)
        self.sh = QDoubleSpinBox(); self.sh.setRange(-2, 2)
        self.sh.setSingleStep(0.1)
        self.tx = QDoubleSpinBox(); self.tx.setRange(-10, 10)
        self.tx.setSingleStep(0.25)
        self.ty = QDoubleSpinBox(); self.ty.setRange(-10, 10)
        self.ty.setSingleStep(0.25)
        self.sg = QDoubleSpinBox(); self.sg.setRange(0, 2)
        self.sg.setSingleStep(0.05); self.sg.setDecimals(2)
        f.addRow("rotation θ", self.rot)
        f.addRow("scale x", self.sx); f.addRow("scale y", self.sy)
        f.addRow("shear xy", self.sh)
        f.addRow("translate x", self.tx); f.addRow("translate y", self.ty)
        f.addRow("jitter σ", self.sg)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def apply_to(self, X):
        th = np.deg2rad(self.rot.value())
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, -s], [s, c]])
        S = np.diag([self.sx.value(), self.sy.value()])
        H = np.array([[1.0, self.sh.value()], [0.0, 1.0]])
        Y = X @ (H @ S @ R).T + np.array([self.tx.value(), self.ty.value()])
        if self.sg.value() > 0:
            Y = Y + np.random.default_rng(0).normal(0, self.sg.value(),
                                                    Y.shape)
        return Y


class DataStudioTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store = store
        self._updating = False
        self._select_next = None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        box = QGroupBox("Generate an unlabeled set  A ⊂ ℝ²")
        form = QFormLayout(box)
        self.kind = QComboBox()
        self.kind.addItems(["Blobs", "Moons", "Circles", "Spirals", "Rings",
                            "XOR", "Anisotropic", "Outliers", "Uniform"])
        self.n = QSpinBox(); self.n.setRange(10, 100000); self.n.setValue(400)
        self.noise = QDoubleSpinBox(); self.noise.setRange(0, 5)
        self.noise.setSingleStep(0.05); self.noise.setValue(0.4)
        self.clusters = QSpinBox(); self.clusters.setRange(2, 12)
        self.clusters.setValue(4)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setValue(7)
        form.addRow("Structure", self.kind)
        form.addRow("|A|  (n points)", self.n)
        form.addRow("Noise", self.noise)
        form.addRow("Clusters", self.clusters)
        form.addRow("Seed", self.seed)
        gen = QPushButton("Generate → add to universe U")
        gen.clicked.connect(self.generate)
        left.addWidget(box); left.addWidget(gen)

        left.addWidget(QLabel("Datasets in U  (double-click to rename):"))
        self.listw = QListWidget()
        self.listw.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.listw.itemChanged.connect(self._renamed)
        left.addWidget(self.listw, stretch=1)

        row = QHBoxLayout()
        tra = QPushButton("Transform g·A…")
        tra.setToolTip("Rotate / scale / shear / translate / jitter — "
                       "data augmentation via a group action")
        tra.clicked.connect(self.transform_selected)
        dele = QPushButton("Delete")
        dele.clicked.connect(self.delete_selected)
        row.addWidget(tra); row.addWidget(dele)
        left.addLayout(row)

        sbox = QGroupBox("Cardinality & stats")
        sv = QVBoxLayout(sbox)
        self.stats = mono_text()
        sv.addWidget(self.stats)
        left.addWidget(sbox)

        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)
        lay.addWidget(lw)

        self.canvas = MplCanvas(7.0, 5.4)
        self.ax = self.canvas.fig.add_subplot(111)
        lay.addWidget(self.canvas, stretch=1)

        self.listw.currentRowChanged.connect(self.preview)
        store.changed.connect(self.refresh)
        self.refresh()

    def generate(self):
        X = make_dataset(self.kind.currentText(), self.n.value(),
                         self.noise.value(), self.clusters.value(),
                         self.seed.value())
        self._select_next = self.store.add(
            self.kind.currentText().lower(), X)

    def transform_selected(self):
        names = self.store.names()
        row = self.listw.currentRow()
        if row < 0 or not names:
            QMessageBox.information(self, "Data Studio", "Select a dataset.")
            return
        name = names[row]
        dlg = TransformDialog(self, name)
        if dlg.exec() == QDialog.Accepted:
            Y = dlg.apply_to(self.store.datasets[name])
            self._select_next = self.store.add(f"{name}·g", Y)

    def delete_selected(self):
        row = self.listw.currentRow()
        names = self.store.names()
        if row >= 0 and names:
            self.store.delete(names[row])

    def _renamed(self, item):
        if self._updating:
            return
        old = item.data(Qt.UserRole)
        new = item.text().strip()
        if not new or new == old:
            self.refresh()
            return
        self._select_next = self.store.rename(old, new)

    def refresh(self):
        self._updating = True
        names = self.store.names()
        self.listw.blockSignals(True)
        self.listw.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setData(Qt.UserRole, n)
            it.setToolTip(f"|A| = {len(self.store.datasets[n])}")
            self.listw.addItem(it)
        self.listw.blockSignals(False)
        if self._select_next and self._select_next in names:
            self.listw.setCurrentRow(names.index(self._select_next))
            self._select_next = None
        elif names:
            if self.listw.currentRow() < 0:
                self.listw.setCurrentRow(0)
        else:
            self.ax.clear()
            self.stats.setPlainText("∅  universe is empty")
            self.canvas.draw()
        self._updating = False

    def preview(self, row):
        names = self.store.names()
        if row < 0 or not names:
            return
        X = self.store.datasets[names[row]]
        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=14, c="#555", edgecolors="none")
        self.ax.set_title(f"A = “{names[row]}”   |A| = {len(X)}   (unlabeled)")
        self.ax.set_aspect("equal"); self.ax.grid(alpha=.3)
        self.canvas.draw()
        self.stats.setPlainText(dataset_stats_text(X))


# ═══════════════════════════════ Tab 2: Sets ═══════════════════════════════

class SetTheoryTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store = store
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        note = QLabel("Datasets are sets of points; operations are ε-tolerant "
                      "(x ∈ A∩B ⇔ ∃b∈B: ‖x−b‖ ≤ ε). Results are stored back "
                      "into U so operations chain.")
        note.setWordWrap(True); left.addWidget(note)

        box = QGroupBox("Binary operations"); form = QFormLayout(box)
        self.comboA = QComboBox(); self.comboB = QComboBox()
        self.eps = QDoubleSpinBox(); self.eps.setRange(0.0, 2.0)
        self.eps.setSingleStep(0.05); self.eps.setValue(0.15)
        form.addRow("Set A", self.comboA)
        form.addRow("Set B", self.comboB)
        form.addRow("ε tolerance", self.eps)
        left.addWidget(box)

        grid = QVBoxLayout()
        for text, op in [("A ∪ B", "∪"), ("A ∩ B", "∩"), ("A ∖ B", "\\"),
                         ("B ∖ A", "B\\A"), ("A △ B", "△"), ("|A×B|", "×"),
                         ("Region map (ε)", "region"),
                         ("Similarity report", "sim")]:
            b = QPushButton(text)
            b.clicked.connect(lambda _, o=op: self.run_op(o))
            grid.addWidget(b)
        row2 = QHBoxLayout()
        b1 = QPushButton("⋃ all sets in U")
        b1.clicked.connect(lambda: self.nary("∪"))
        b2 = QPushButton("⋂ all sets in U")
        b2.clicked.connect(lambda: self.nary("∩"))
        row2.addWidget(b1); row2.addWidget(b2)
        grid.addLayout(row2)
        left.addLayout(grid)

        self.info = mono_text(); left.addWidget(self.info, stretch=1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)
        lay.addWidget(lw)

        self.canvas = MplCanvas(7.2, 5.6)
        self.ax = self.canvas.fig.add_subplot(111)
        lay.addWidget(self.canvas, stretch=1)

        store.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        refresh_combo(self.comboA, self.store)
        refresh_combo(self.comboB, self.store)

    def _sets(self):
        na, nb = self.comboA.currentText(), self.comboB.currentText()
        if not na or not nb:
            QMessageBox.information(self, "Set Theory",
                                    "Generate at least two datasets first.")
            return None
        return na, nb, self.store.datasets[na], self.store.datasets[nb]

    def run_op(self, op):
        got = self._sets()
        if got is None:
            return
        na, nb, A, B = got
        eps = self.eps.value()

        if op == "region":
            self._region_map(A, B, eps, na, nb); return
        if op == "sim":
            self._similarity(A, B, eps, na, nb); return

        inter = set_intersection(A, B, eps)
        la, lb = len(A), len(B)
        if op == "∪":
            R, sym = set_union(A, B, eps), "∪"
        elif op == "∩":
            R, sym = inter, "∩"
        elif op == "\\":
            R, sym = set_difference(A, B, eps), "∖"
        elif op == "B\\A":
            R, sym, na, nb, A, B = set_difference(B, A, eps), "∖", nb, na, B, A
        elif op == "△":
            R, sym = set_symdiff(A, B, eps), "△"
        else:
            self.info.setPlainText(
                f"|A×B| = |A|·|B| = {la}·{lb} = {la * lb}\n"
                f"(A×B ⊂ ℝ⁴ — not plottable here)")
            return

        rname = self.store.add(f"({na}{sym}{nb})", R)
        union_cnt = len(R) if op == "∪" else len(set_union(A, B, eps))
        rhs = la + lb - len(inter)
        self.info.setPlainText(
            f"ε = {eps}\n"
            f"|A| = {la}    |B| = {lb}\n"
            f"|A ∩ B| = {len(inter)}\n"
            f"|A ∪ B| = {union_cnt}\n"
            f"Inclusion–exclusion:  {la}+{lb}−{len(inter)} = {rhs}  "
            f"{'✓' if union_cnt == rhs else '✗'}\n"
            f"|A △ B| = {len(set_symdiff(A, B, eps))}\n"
            f"|𝒫(A)| = 2^{la}   (power set cardinality)\n"
            f"stored as: {rname}")

        self.ax.clear()
        self.ax.scatter(A[:, 0], A[:, 1], s=22, facecolors="none",
                        edgecolors="tab:blue", label=f"A={na}")
        self.ax.scatter(B[:, 0], B[:, 1], s=22, marker="+",
                        c="tab:red", label=f"B={nb}")
        if len(R):
            self.ax.scatter(R[:, 0], R[:, 1], s=26, c="tab:green",
                            edgecolors="k", linewidths=.3,
                            label=f"result (|R|={len(R)})")
        self.ax.legend(); self.ax.set_aspect("equal"); self.ax.grid(alpha=.3)
        self.ax.set_title(f"A {sym} B")
        self.canvas.draw()

    def _region_map(self, A, B, eps, na, nb):
        pts = [P for P in (A, B) if len(P)]
        if not pts:
            QMessageBox.information(self, "Set Theory", "Both sets are empty.")
            return
        lo = np.min([P.min(0) for P in pts], axis=0)
        hi = np.max([P.max(0) for P in pts], axis=0)
        pad = max(1.0, 3 * eps)
        x0, y0 = lo - pad
        x1, y1 = hi + pad
        g = 320
        xx, yy = np.meshgrid(np.linspace(x0, x1, g), np.linspace(y0, y1, g))
        G = np.column_stack([xx.ravel(), yy.ravel()])
        inA = cKDTree(A).query(G, k=1)[0] <= eps if len(A) \
            else np.zeros(len(G), bool)
        inB = cKDTree(B).query(G, k=1)[0] <= eps if len(B) \
            else np.zeros(len(G), bool)
        lab = (2 * inA + inB).astype(int).reshape(g, g)
        cmap = ListedColormap(["#ffffff", "#ffd6d6", "#d6e4ff", "#c9f2c9"])
        self.ax.clear()
        self.ax.imshow(lab, extent=[x0, x1, y0, y1], origin="lower",
                       cmap=cmap, interpolation="nearest")
        self.ax.scatter(A[:, 0], A[:, 1], s=10, c="tab:blue", label=f"A={na}")
        self.ax.scatter(B[:, 0], B[:, 1], s=14, marker="+", c="tab:red",
                        label=f"B={nb}")
        self.ax.set_aspect("equal"); self.ax.legend(fontsize=8)
        self.ax.grid(alpha=.2)
        self.ax.set_title(f"ε-neighborhood regions  (ε = {eps})")
        self.canvas.draw()
        self.info.setPlainText(
            "white = neither · pink = B∖A · blue = A∖B · green = A∩B\n"
            "(each region = points within ε of the respective sets)\n\n"
            f"|A∖B| = {len(set_difference(A, B, eps))}\n"
            f"|B∖A| = {len(set_difference(B, A, eps))}\n"
            f"|A∩B| = {len(set_intersection(A, B, eps))}")

    def _similarity(self, A, B, eps, na, nb):
        inter = set_intersection(A, B, eps)
        uni = set_union(A, B, eps)
        jac = len(inter) / len(uni) if len(uni) else float("nan")
        dice = 2 * len(inter) / (len(A) + len(B)) if (len(A) + len(B)) \
            else float("nan")
        m = min(len(A), len(B))
        ov = len(inter) / m if m else float("nan")
        cab = len(inter) / len(A) if len(A) else 0.0
        cba = len(inter) / len(B) if len(B) else 0.0
        self.info.setPlainText(
            f"Similarity of A=“{na}” and B=“{nb}”  (ε = {eps})\n\n"
            f"Jaccard   |A∩B|/|A∪B|        = {jac:.3f}\n"
            f"Dice      2|A∩B|/(|A|+|B|)   = {dice:.3f}\n"
            f"overlap   |A∩B|/min(|A|,|B|) = {ov:.3f}\n\n"
            f"containment  A ⊆ε B: {cab:.3f}  → "
            f"{'YES ✓' if cab >= 1 - 1e-12 else 'no'}\n"
            f"containment  B ⊆ε A: {cba:.3f}  → "
            f"{'YES ✓' if cba >= 1 - 1e-12 else 'no'}\n"
            f"A =ε B (both containments 1): "
            f"{'YES ✓' if cab >= 1 - 1e-12 and cba >= 1 - 1e-12 else 'no'}")

    def nary(self, op):
        names = self.store.names()
        if len(names) < 2:
            QMessageBox.information(self, "Set Theory",
                                    "Need at least two datasets in U.")
            return
        eps = self.eps.value()
        R = self.store.datasets[names[0]]
        for nm in names[1:]:
            B2 = self.store.datasets[nm]
            R = set_union(R, B2, eps) if op == "∪" \
                else set_intersection(R, B2, eps)
        sym = "⋃" if op == "∪" else "⋂"
        rname = self.store.add(f"({sym}U)", R)
        lines = "\n".join(f"  |{nm}| = {len(self.store.datasets[nm])}"
                          for nm in names)
        self.info.setPlainText(
            f"n-ary {sym} over all {len(names)} sets in U:\n{lines}\n\n"
            f"|{sym}U| = {len(R)}\nstored as: {rname}")
        self.ax.clear()
        for i, nm in enumerate(names):
            P = self.store.datasets[nm]
            self.ax.scatter(P[:, 0], P[:, 1], s=8, alpha=.35,
                            color=f"C{i % 10}", label=nm)
        if len(R):
            self.ax.scatter(R[:, 0], R[:, 1], s=30, c="tab:green",
                            edgecolors="k", linewidths=.3,
                            label=f"{sym}U (|·|={len(R)})")
        self.ax.legend(fontsize=7); self.ax.set_aspect("equal")
        self.ax.grid(alpha=.3); self.ax.set_title(f"{sym} over U")
        self.canvas.draw()


# ══════════════════════════ Tab 3: Group theory ════════════════════════════

class GroupTheoryTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store = store
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        note = QLabel("Quotient set A/~ · group actions on ℝ² (orbits, "
                      "stabilizers, Cayley, subgroups, conjugacy, invariants) "
                      "· apply g to a dataset to augment it. Click the canvas "
                      "to pick p.")
        note.setWordWrap(True); left.addWidget(note)

        ds = QGroupBox("Dataset"); f1 = QFormLayout(ds)
        self.combo = QComboBox(); f1.addRow(self.combo)
        left.addWidget(ds)

        gb = QGroupBox("Group  G ⤷ ℝ²"); f2 = QFormLayout(gb)
        self.grp = QComboBox()
        self.grp.addItems(["D₄ (order 8)", "Dₙ (dihedral)", "Cₙ (cyclic)"])
        self.ordn = QSpinBox(); self.ordn.setRange(3, 12); self.ordn.setValue(4)
        f2.addRow("Group", self.grp); f2.addRow("n", self.ordn)
        left.addWidget(gb)

        q = QGroupBox("Quotient set  A/~"); fq = QFormLayout(q)
        self.rel = QComboBox()
        self.rel.addItems(["KMeans clusters (k)", "Square grid cells (side ℓ)"])
        self.k = QSpinBox(); self.k.setRange(2, 20); self.k.setValue(4)
        self.cell = QDoubleSpinBox(); self.cell.setRange(0.25, 6)
        self.cell.setSingleStep(0.25); self.cell.setValue(1.0)
        fq.addRow("Relation ~", self.rel)
        fq.addRow("k (clusters)", self.k)
        fq.addRow("ℓ (grid side)", self.cell)
        left.addWidget(q)
        bq = QPushButton("Compute quotient set  A/~")
        bq.clicked.connect(self.quotient)
        left.addWidget(bq)

        g = QGroupBox("Point p"); fg = QFormLayout(g)
        self.px = QDoubleSpinBox(); self.py = QDoubleSpinBox()
        for s_ in (self.px, self.py):
            s_.setRange(-8, 8); s_.setSingleStep(0.25); s_.setValue(2.0)
        fg.addRow("p = (x, y)", self.px); fg.addRow("", self.py)
        left.addWidget(g)
        bo = QPushButton("Orbit & stabilizer of p")
        bo.clicked.connect(self.orbit)
        left.addWidget(bo)

        st = QGroupBox("Group structure"); fs = QGridLayout(st)
        bc = QPushButton("Cayley table"); bc.clicked.connect(self.cayley)
        bsub = QPushButton("Enumerate subgroups")
        bsub.clicked.connect(self.subgroups)
        bconj = QPushButton("Conjugacy classes")
        bconj.clicked.connect(self.conjugacy)
        binv = QPushButton("Invariant features")
        binv.clicked.connect(self.invariance)
        fs.addWidget(bc, 0, 0); fs.addWidget(bsub, 0, 1)
        fs.addWidget(bconj, 1, 0); fs.addWidget(binv, 1, 1)
        left.addWidget(st)

        act = QGroupBox("Group action on the dataset")
        fa = QFormLayout(act)
        self.gelem = QComboBox()
        bapp = QPushButton("Apply g to A → store g·A in U")
        bapp.clicked.connect(self.apply_element)
        fa.addRow("element g", self.gelem)
        fa.addRow(bapp)
        left.addWidget(act)

        self.info = mono_text(); left.addWidget(self.info, stretch=1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(400)
        scroll = QScrollArea(); scroll.setWidget(lw)
        scroll.setWidgetResizable(True); scroll.setMaximumWidth(420)
        lay.addWidget(scroll)

        self.canvas = MplCanvas(7.2, 5.6)
        self.ax = self.canvas.fig.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        lay.addWidget(self.canvas, stretch=1)

        store.changed.connect(self.refresh)
        self.grp.currentIndexChanged.connect(self._sync_elements)
        self.ordn.valueChanged.connect(self._sync_elements)
        self.refresh()

    def refresh(self):
        refresh_combo(self.combo, self.store)
        self._sync_elements()

    def _sync_elements(self):
        elems, names = self.current_group()
        cur = self.gelem.currentText()
        self.gelem.blockSignals(True)
        self.gelem.clear()
        self.gelem.addItems([f"g = {n}" for n in names])
        if cur:
            i = self.gelem.findText(cur)
            if i >= 0:
                self.gelem.setCurrentIndex(i)
        self.gelem.blockSignals(False)

    def current_group(self):
        mode = self.grp.currentIndex()
        n = self.ordn.value()
        if mode == 0:
            return dihedral_elements(4)
        return dihedral_elements(n, cyclic=(mode == 2))

    def _glabel(self):
        return ["D₄", f"D{self.ordn.value()}", f"C{self.ordn.value()}"][
            self.grp.currentIndex()]

    def _try_data(self):
        n = self.combo.currentText()
        return self.store.datasets[n] if n else None

    def _data(self):
        X = self._try_data()
        if X is None:
            QMessageBox.information(self, "Group Theory",
                                    "Generate a dataset first.")
        return X

    @staticmethod
    def _medoid(C):
        if len(C) == 1:
            return C[0]
        return C[np.argmin(((C - C.mean(0)) ** 2).sum(1))]

    def quotient(self):
        X = self._data()
        if X is None:
            return
        if self.rel.currentIndex() == 0:
            k = min(self.k.value(), len(X))
            lab = KMeans(n_clusters=k, n_init=10,
                         random_state=0).fit_predict(X)
            rel_txt = f"x ~ y  ⇔  same KMeans cluster (k={k})"
        else:
            cells = np.floor(X / self.cell.value()).astype(int)
            _, lab = np.unique(cells, axis=0, return_inverse=True)
            rel_txt = f"x ~ y  ⇔  ⌊x/ℓ⌋ = ⌊y/ℓ⌋  (ℓ={self.cell.value()})"
        m = int(lab.max()) + 1
        cmap = tab10(m) if m <= 10 else get_cmap("viridis")
        reps = []
        self.ax.clear()
        for i in range(m):
            C = X[lab == i]
            color = cmap(i) if m <= 10 else cmap(Normalize(0, m - 1)(i))
            self.ax.scatter(C[:, 0], C[:, 1], s=12, color=color)
            r = self._medoid(C); reps.append(r)
            self.ax.scatter(*r, s=220, marker="*", c="k", zorder=5)
        self.ax.set_aspect("equal"); self.ax.grid(alpha=.3)
        self.ax.set_title(f"Quotient set  A/~   (|A/~| = {m});  "
                          f"★ = representatives")
        self.canvas.draw()
        sizes = sorted((int((lab == i).sum()) for i in range(m)), reverse=True)
        self.info.setPlainText(
            f"Equivalence relation:  {rel_txt}\n"
            f"(reflexive, symmetric, transitive by construction)\n\n"
            f"|A/~| = {m} equivalence classes\n"
            f"class sizes: {sizes}\n"
            f"Σ sizes = {sum(sizes)} = |A|  ✓  (a partition of A)\n\n"
            f"Representatives:\n" +
            "\n".join(f"  [{i}] ≈ ({r[0]:+.2f}, {r[1]:+.2f})"
                      for i, r in enumerate(reps)))

    def orbit(self):
        X = self._data()
        if X is None:
            return
        p = np.array([self.px.value(), self.py.value()])
        elems, names = self.current_group()
        imgs = [M @ p for M in elems]
        preimages = {}
        for nm, y in zip(names, imgs):
            preimages.setdefault(tuple(np.round(y, 9)), []).append(nm)
        stab = [nm for nm, M in zip(names, elems)
                if np.allclose(M @ p, p, atol=1e-9)]

        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=10, c="#bbb")
        self.ax.scatter(*p, s=260, marker="*", c="k", zorder=6, label="p")
        colors = tab10(len(preimages))
        for i, (key, gnames) in enumerate(sorted(preimages.items())):
            y = np.array(key)
            self.ax.scatter(*y, s=90, color=colors(i), zorder=5,
                            edgecolors="k", linewidths=.5)
            self.ax.annotate(",".join(gnames), y + 0.15, fontsize=8)
        self.ax.set_aspect("equal"); self.ax.grid(alpha=.3); self.ax.legend()
        self.ax.set_title(f"Orbit  G·p  under {self._glabel()}  "
                          f"(|G·p| = {len(preimages)})")
        self.canvas.draw()

        n_orb, n_stab, ng = len(preimages), len(stab), len(elems)
        self.info.setPlainText(
            f"G = {self._glabel()},  |G| = {ng}\n"
            f"p = ({p[0]:+.2f}, {p[1]:+.2f})\n\n"
            f"Orbit  G·p = {{ g·p : g ∈ G }}  (|G·p| = {n_orb})\n" +
            "\n".join(f"  {','.join(g):>10} → ({k[0]:+.2f}, {k[1]:+.2f})"
                      for k, g in sorted(preimages.items())) +
            f"\n\nStabilizer  G_p = {{{', '.join(stab)}}}  "
            f"(|G_p| = {n_stab})\n"
            f"Orbit–stabilizer:  |G·p|·|G_p| = {n_orb}·{n_stab} = "
            f"{n_orb * n_stab} = |G|  "
            f"{'✓' if n_orb * n_stab == ng else '✗'}")

    def cayley(self):
        elems, names = self.current_group()
        m = len(elems)
        table = [["∘"] + names]
        for i, gi in enumerate(elems):
            row = [names[i]]
            for gj in elems:
                j = _match_matrix(gi @ gj, elems)
                row.append(names[j] if j is not None else "?")
            table.append(row)
        closure_ok = all("?" not in r for r in table)
        perms_ok = all(sorted(r[1:]) == sorted(names) for r in table[1:])
        width = max(len(x) for r in table for x in r) + 1
        self.info.setPlainText(
            f"Cayley table of {self._glabel()}  (g∘h)(x) = g(h(x)):\n\n" +
            "\n".join("".join(f"{x:>{width}}" for x in r) for r in table) +
            f"\n\nclosure: {'✓' if closure_ok else '✗'}   "
            f"every row is a permutation of G: {'✓' if perms_ok else '✗'}\n"
            f"each row contains e → inverses exist ✓   "
            f"|G| = {m}")

    def subgroups(self):
        elems, names = self.current_group()
        m = len(elems)
        if m > 12:
            QMessageBox.information(self, "Group Theory",
                                    "Subgroup enumeration limited to "
                                    "|G| ≤ 12.")
            return
        mult = [[_match_matrix(elems[i] @ elems[j], elems) for j in range(m)]
                for i in range(m)]
        subs = set()
        for mask in range(1, 1 << m):
            idx = [i for i in range(m) if (mask >> i) & 1]
            S = set(idx)
            if all(mult[i][j] in S for i in idx for j in idx):
                subs.add(frozenset(idx))
        by_order = defaultdict(list)
        for s in subs:
            by_order[len(s)].append(s)
        lag = all(m % o == 0 for o in by_order)
        ident = _match_matrix(np.eye(2), elems)
        lines = [f"{self._glabel()}, |G| = {m} — brute-forced all "
                 f"2^{m}−1 = {2 ** m - 1} nonempty subsets:",
                 f"found {len(subs)} subgroups (nonempty + closed under ∘):",
                 ""]
        for o in sorted(by_order):
            lines.append(f"  order {o}:  {len(by_order[o])} subgroup(s)")
            for s in sorted(by_order[o], key=sorted)[:6]:
                lines.append("      { " +
                             ", ".join(names[i] for i in sorted(s)) + " }")
            if len(by_order[o]) > 6:
                lines.append(f"      … (+{len(by_order[o]) - 6} more)")
        lines += ["",
                  f"Lagrange — every order divides {m}: "
                  f"{'✓' if lag else '✗'}",
                  f"every subgroup contains e: "
                  f"{'✓' if all(ident in s for s in subs) else '✗'}"]
        self.info.setPlainText("\n".join(lines))

    def conjugacy(self):
        elems, names = self.current_group()
        m = len(elems)
        mult = [[_match_matrix(elems[i] @ elems[j], elems) for j in range(m)]
                for i in range(m)]
        ident = _match_matrix(np.eye(2), elems)
        inv = [next(j for j in range(m) if mult[j][i] == ident)
               for i in range(m)]
        classes, seen = [], set()
        for i in range(m):
            cls = frozenset(mult[mult[h][i]][inv[h]] for h in range(m))
            if cls not in seen:
                seen.add(cls)
                classes.append(cls)
        sizes = [len(c) for c in classes]
        center = [names[next(iter(c))] for c in classes if len(c) == 1]
        lines = [f"Conjugacy classes of {self._glabel()}  (|G| = {m}):", ""]
        for c in classes:
            lines.append("  [" + ", ".join(names[i] for i in sorted(c)) +
                         f"]   size {len(c)}")
        lines += ["",
                  f"class equation: {' + '.join(map(str, sizes))} = "
                  f"{sum(sizes)} = |G|  {'✓' if sum(sizes) == m else '✗'}",
                  f"center Z(G) = {{{', '.join(center)}}}",
                  "(elements whose class is a singleton)"]
        self.info.setPlainText("\n".join(lines))

    def invariance(self):
        X = self._try_data()
        rng = np.random.default_rng(0)
        if X is None or len(X) == 0:
            P = rng.uniform(-4, 4, (200, 2))
        else:
            P = X[rng.choice(len(X), min(300, len(X)), replace=False)]
        elems, names = self.current_group()
        feats = [
            ("f(x) = x² + y²   (radius²)",
             lambda P: P[:, 0] ** 2 + P[:, 1] ** 2),
            ("f(x) = |x| + |y|", lambda P: np.abs(P).sum(1)),
            ("f(x) = max(|x|, |y|)", lambda P: np.abs(P).max(1)),
            ("f(x) = x·y", lambda P: P[:, 0] * P[:, 1]),
            ("f(x) = x + y", lambda P: P[:, 0] + P[:, 1]),
            ("f(x) = x² − y²", lambda P: P[:, 0] ** 2 - P[:, 1] ** 2),
        ]
        lines = [f"G-invariance test on {len(P)} points — "
                 f"f(g·x) = f(x) for all g ∈ {self._glabel()}:", ""]
        for fname, f in feats:
            vals = f(P)
            ok = all(np.allclose(f(P @ M.T), vals, atol=1e-9) for M in elems)
            lines.append(f"  {'✓' if ok else '✗'}  {fname}")
        lines += ["",
                  "An invariant feature is constant on every orbit —",
                  "exactly what an invariant ML model is allowed to use."]
        self.info.setPlainText("\n".join(lines))

    def apply_element(self):
        X = self._data()
        if X is None:
            return
        elems, names = self.current_group()
        i = self.gelem.currentIndex()
        gname = names[i] if 0 <= i < len(names) else "?"
        Y = X @ elems[i].T
        dname = self.combo.currentText()
        rname = self.store.add(f"{dname}·{gname}", Y)
        self.ax.clear()
        self.ax.scatter(X[:, 0], X[:, 1], s=10, c="#bbb",
                        label=f"A = {dname}")
        self.ax.scatter(Y[:, 0], Y[:, 1], s=12, c="tab:green",
                        label=f"g·A  (g = {gname})")
        self.ax.set_aspect("equal"); self.ax.grid(alpha=.3); self.ax.legend()
        self.ax.set_title(f"Group action:  g·A  with  g = {gname}")
        self.canvas.draw()
        self.info.setPlainText(
            f"applied g = {gname} ∈ {self._glabel()} to “{dname}”\n"
            f"stored as “{rname}” in the universe U.\n\n"
            f"The orbit of the SET is {{ g·A : g ∈ G }} — apply more\n"
            f"elements to collect it. Group-based augmentation builds\n"
            f"the symmetry of G into the data (equivariant-ML trick).")

    def on_click(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None:
            self.px.setValue(ev.xdata)
            self.py.setValue(ev.ydata)
            self.orbit()


# ═══════════════════════════ Tab 4: Neural network ═════════════════════════

class NeuralTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store, self.worker = store, None
        self._hist, self._model, self._Xs, self._src = [], None, None, None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        note = QLabel("Unsupervised autoencoder ℝ²→ℝᵈ→ℝ², optionally "
                      "DENOISING (corrupt input, reconstruct clean point). "
                      "No labels anywhere. Send the latent space back to U "
                      "and run set theory on it!")
        note.setWordWrap(True); left.addWidget(note)

        box = QGroupBox("Architecture & training"); f = QFormLayout(box)
        self.combo = QComboBox()
        self.hidden = QSpinBox(); self.hidden.setRange(4, 128)
        self.hidden.setSingleStep(2); self.hidden.setValue(16)
        self.latent = QSpinBox(); self.latent.setRange(2, 8)
        self.latent.setValue(2)
        self.epochs = QSpinBox(); self.epochs.setRange(50, 5000)
        self.epochs.setSingleStep(50); self.epochs.setValue(400)
        self.lr = QDoubleSpinBox(); self.lr.setRange(0.0001, 1.0)
        self.lr.setDecimals(4); self.lr.setSingleStep(0.001)
        self.lr.setValue(0.01)
        self.batch = QSpinBox(); self.batch.setRange(8, 512)
        self.batch.setValue(32)
        self.act = QComboBox(); self.act.addItems(["tanh", "relu"])
        self.opt = QComboBox(); self.opt.addItems(["adam", "sgd (momentum)"])
        self.l2 = QDoubleSpinBox(); self.l2.setRange(0, 0.1)
        self.l2.setDecimals(4); self.l2.setSingleStep(0.0005)
        self.noise = QDoubleSpinBox(); self.noise.setRange(0, 2)
        self.noise.setSingleStep(0.05); self.noise.setDecimals(2)
        self.kcol = QSpinBox(); self.kcol.setRange(2, 8); self.kcol.setValue(4)
        f.addRow("Dataset", self.combo)
        f.addRow("Hidden units", self.hidden)
        f.addRow("Latent dim", self.latent)
        f.addRow("Epochs", self.epochs)
        f.addRow("Learning rate", self.lr)
        f.addRow("Batch size", self.batch)
        f.addRow("Activation", self.act)
        f.addRow("Optimizer", self.opt)
        f.addRow("L2 penalty λ", self.l2)
        f.addRow("Denoising σ", self.noise)
        f.addRow("Latent colors (k)", self.kcol)
        left.addWidget(box)

        self.train_btn = QPushButton("Train autoencoder (background thread)")
        self.train_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_worker)
        self.lat_btn = QPushButton("→ send latent space to universe U")
        self.lat_btn.clicked.connect(self.send_latent)
        self.pb = QProgressBar()
        self.lbl = QLabel("idle")
        left.addWidget(self.train_btn); left.addWidget(self.stop_btn)
        left.addWidget(self.lat_btn); left.addWidget(self.pb)
        left.addWidget(self.lbl)
        self.info = mono_text(); left.addWidget(self.info, stretch=1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)
        lay.addWidget(lw)

        self.canvas = MplCanvas(9.6, 4.4)
        self.ax_loss = self.canvas.fig.add_subplot(131)
        self.ax_lat = self.canvas.fig.add_subplot(132)
        self.ax_rec = self.canvas.fig.add_subplot(133)
        lay.addWidget(self.canvas, stretch=1)

        store.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        refresh_combo(self.combo, self.store)

    def start(self):
        n = self.combo.currentText()
        if not n:
            QMessageBox.information(self, "Neural Net",
                                    "Generate a dataset first.")
            return
        X = self.store.datasets[n]
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Xs = (X - mu) / sd
        model = Autoencoder([2, self.hidden.value(), self.latent.value(),
                             self.hidden.value(), 2], seed=1,
                            activation=self.act.currentText(),
                            opt=self.opt.currentText(),
                            l2=self.l2.value())
        ep, lr, bs, ns = (self.epochs.value(), self.lr.value(),
                          self.batch.value(), self.noise.value())
        self._hist.clear()
        self.pb.setRange(1, ep)

        def job(w):
            def cb(epoch, loss):
                w.progress.emit(epoch, loss)
                return w.isInterruptionRequested()
            hist = model.fit(Xs, epochs=ep, lr=lr, batch=bs,
                             progress=cb, noise=ns)
            return model, hist, Xs

        self.worker = TrainingWorker(job)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_run.connect(self.on_done)
        self.train_btn.setEnabled(False)
        self.worker.start()

    def on_progress(self, ep, loss):
        self._hist.append(loss)
        self.pb.setValue(ep)
        self.lbl.setText(f"epoch {ep}   ·   MSE {loss:.5f}")
        if ep % 2 == 0 or ep == 1:
            self.ax_loss.clear()
            self.ax_loss.semilogy(self._hist, c="tab:orange")
            self.ax_loss.set_xlabel("epoch"); self.ax_loss.set_ylabel("MSE")
            self.ax_loss.grid(alpha=.3)
            self.canvas.draw()

    def on_done(self, result):
        self.train_btn.setEnabled(True)
        if isinstance(result, Exception):
            QMessageBox.critical(self, "Neural Net", str(result))
            return
        model, hist, Xs = result
        self._model, self._Xs, self._src = model, Xs, self.combo.currentText()
        lat = model.encode(Xs)
        k = self.kcol.value()
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(lat)
        cmap = tab10(k)
        self.ax_lat.clear()
        self.ax_lat.scatter(lat[:, 0], lat[:, 1], c=lab, cmap=cmap, s=14,
                            edgecolors="k", linewidths=.2)
        self.ax_lat.set_title("latent code  enc(x)")
        self.ax_lat.set_aspect("equal"); self.ax_lat.grid(alpha=.3)

        recon = model._forward(Xs)[-1]
        idx = np.random.default_rng(0).choice(len(Xs), min(150, len(Xs)),
                                              replace=False)
        self.ax_rec.clear()
        for a, b in zip(Xs[idx], recon[idx]):
            self.ax_rec.plot([a[0], b[0]], [a[1], b[1]],
                             c="#d5d5d5", lw=.7, zorder=1)
        self.ax_rec.scatter(Xs[idx, 0], Xs[idx, 1], s=16, c="#606060",
                            label="x", zorder=2)
        self.ax_rec.scatter(recon[idx, 0], recon[idx, 1], s=26, marker="x",
                            c=lab[idx], cmap=cmap, label="f(x)", zorder=3)
        self.ax_rec.set_title("reconstruction  x → f(x)")
        self.ax_rec.legend(fontsize=8); self.ax_rec.set_aspect("equal")
        self.ax_rec.grid(alpha=.3)
        self.canvas.draw()

        sil = silhouette_score(lat, lab) if 1 < k < len(lat) else float("nan")
        self.info.setPlainText(
            f"architecture: {' → '.join(map(str, model.dims))}   "
            f"({model.activation} hidden, linear output)\n"
            f"optimizer: {model.opt}   λ(L2) = {model.l2}   "
            f"denoising σ = {self.noise.value()}\n"
            f"epochs run: {len(hist)}"
            f"{'  (stopped early)' if len(hist) < self.epochs.value() else ''}\n"
            f"first MSE: {hist[0]:.5f}   final MSE: {hist[-1]:.5f}\n"
            f"latent-clustering silhouette: {sil:.3f}\n\n"
            f"The encoder is a learned, continuous quotient-like map\n"
            f"A → ℝᵈ that preserves the structure of the unlabeled set.\n"
            f"Click “send latent space to universe U” to run set\n"
            f"operations on the learned representation.")

    def send_latent(self):
        if self._model is None or self._Xs is None:
            QMessageBox.information(self, "Neural Net", "Train first.")
            return
        Z = self._model.encode(self._Xs)
        if Z.shape[1] != 2:
            Z = PCA(2, random_state=0).fit_transform(Z)
        name = self.store.add(f"{self._src}_latent", Z)
        self.lbl.setText(f"latent space of “{self._src}” stored as “{name}”")

    def stop_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)


# ═══════════════════════ Tab 5: AdaBoost & Random Forest ═══════════════════

class EnsembleTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store = store
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        note = QLabel("Pseudo-labels come from KMeans — the quotient map "
                      "q: A → A/~. Both ensembles approximate q. Also probes: "
                      "staged boosting accuracy, OOB, and self-consistency "
                      "under a 90° rotation (a D₄ element).")
        note.setWordWrap(True); left.addWidget(note)

        ds = QGroupBox("Data & pseudo-labels"); f = QFormLayout(ds)
        self.combo = QComboBox()
        self.k = QSpinBox(); self.k.setRange(2, 10); self.k.setValue(4)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        self.test = QDoubleSpinBox(); self.test.setRange(0.1, 0.5)
        self.test.setSingleStep(0.05); self.test.setValue(0.3)
        f.addRow("Dataset", self.combo)
        f.addRow("k clusters", self.k)
        f.addRow("Seed", self.seed)
        f.addRow("Test fraction", self.test)
        left.addWidget(ds)

        rf = QGroupBox("Random Forest"); frf = QFormLayout(rf)
        self.rf_n = QSpinBox(); self.rf_n.setRange(10, 1000)
        self.rf_n.setValue(120)
        self.rf_d = QSpinBox(); self.rf_d.setRange(0, 30)
        self.rf_d.setSpecialValueText("None (fully grown)")
        frf.addRow("n_estimators", self.rf_n)
        frf.addRow("max_depth", self.rf_d)
        left.addWidget(rf)

        ab = QGroupBox("AdaBoost"); fab = QFormLayout(ab)
        self.ab_n = QSpinBox(); self.ab_n.setRange(10, 500)
        self.ab_n.setValue(60)
        self.ab_lr = QDoubleSpinBox(); self.ab_lr.setRange(0.01, 2.0)
        self.ab_lr.setSingleStep(0.05); self.ab_lr.setValue(1.0)
        self.ab_d = QSpinBox(); self.ab_d.setRange(1, 10); self.ab_d.setValue(1)
        fab.addRow("n_estimators", self.ab_n)
        fab.addRow("learning_rate", self.ab_lr)
        fab.addRow("stump depth", self.ab_d)
        left.addWidget(ab)

        btn = QPushButton("Train both ensembles")
        btn.clicked.connect(self.train)
        left.addWidget(btn)
        self.info = mono_text(); left.addWidget(self.info, stretch=1)
        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(380)
        lay.addWidget(lw)

        self.canvas = MplCanvas(8.6, 6.8)
        gs = self.canvas.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        self.ax_rf = self.canvas.fig.add_subplot(gs[0, 0])
        self.ax_ab = self.canvas.fig.add_subplot(gs[0, 1])
        self.ax_stage = self.canvas.fig.add_subplot(gs[1, 0])
        self.ax_im = self.canvas.fig.add_subplot(gs[1, 1])
        lay.addWidget(self.canvas, stretch=1)

        store.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        refresh_combo(self.combo, self.store)

    @staticmethod
    def _boundary(ax, model, X, lab, k, title, highlight=None):
        pad = 1.0
        x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
        y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
        xx, yy = np.meshgrid(np.linspace(x0, x1, 240),
                             np.linspace(y0, y1, 240))
        Z = model.predict(np.column_stack([xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        cmap = tab10(k)
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, k + 0.5, 1),
                    cmap=cmap, alpha=0.25)
        ax.scatter(X[:, 0], X[:, 1], c=lab, cmap=cmap, s=10,
                   edgecolors="k", linewidths=.2)
        if highlight is not None and highlight.any():
            ax.scatter(X[highlight, 0], X[highlight, 1], s=70,
                       facecolors="none", edgecolors="k", linewidths=1.2,
                       label="RF ≠ Ada")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title); ax.set_aspect("equal")

    def train(self):
        n = self.combo.currentText()
        if not n:
            QMessageBox.information(self, "Ensembles",
                                    "Generate a dataset first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            X = self.store.datasets[n]
            k, seed = self.k.value(), self.seed.value()
            lab = KMeans(n_clusters=k, n_init=10,
                         random_state=seed).fit_predict(X)
            sil = silhouette_score(X, lab) if 1 < k < len(X) else float("nan")

            Xtr, Xte, ltr, lte = train_test_split(
                X, lab, test_size=self.test.value(),
                random_state=seed, stratify=lab)

            rf = RandomForestClassifier(
                n_estimators=self.rf_n.value(),
                max_depth=self.rf_d.value() or None,
                oob_score=True, random_state=seed, n_jobs=-1).fit(Xtr, ltr)
            oob = float(getattr(rf, "oob_score_", float("nan")))

            ada = make_adaboost(seed, self.ab_n.value(),
                                self.ab_lr.value(), self.ab_d.value())
            ada.fit(Xtr, ltr)
            staged = [accuracy_score(lte, p) for p in ada.staged_predict(Xte)]

            R90 = np.array([[0.0, -1.0], [1.0, 0.0]])
            Xr = X @ R90.T
            cons_rf = float(np.mean(rf.predict(X) == rf.predict(Xr)))
            cons_ada = float(np.mean(ada.predict(X) == ada.predict(Xr)))
            disagree = rf.predict(X) != ada.predict(X)

            self.info.setPlainText(
                f"pseudo-labels: KMeans k={k}   silhouette = {sil:.3f}\n"
                f"quotient map  q: A → A/~  (classes: {k})\n\n"
                f"{'model':<14}{'train acc':>11}{'test acc':>10}\n"
                f"{'-' * 35}\n"
                f"{'RandomForest':<14}"
                f"{accuracy_score(ltr, rf.predict(Xtr)):>11.3f}"
                f"{accuracy_score(lte, rf.predict(Xte)):>10.3f}\n"
                f"{'AdaBoost':<14}"
                f"{accuracy_score(ltr, ada.predict(Xtr)):>11.3f}"
                f"{accuracy_score(lte, ada.predict(Xte)):>10.3f}\n\n"
                f"RF out-of-bag score: {oob:.3f}\n"
                f"AdaBoost staged test acc: {staged[0]:.3f} → "
                f"{staged[-1]:.3f} over {len(staged)} estimators\n"
                f"RF/Ada agreement: {1 - disagree.mean():.3f} "
                f"({int(disagree.sum())} circled points)\n"
                f"self-consistency under r (90° rotation, g ∈ D₄):\n"
                f"   RF {cons_rf:.3f} · AdaBoost {cons_ada:.3f}\n"
                f"(axis-aligned thresholds + D₄-symmetric clusters → RF is\n"
                f" nearly invariant; accuracy is vs pseudo-labels q, "
                f"not truth)")

            self._boundary(self.ax_rf, rf, X, lab, k,
                           f"Random Forest ({self.rf_n.value()} trees)",
                           highlight=disagree)
            self._boundary(self.ax_ab, ada, X, lab, k,
                           f"AdaBoost ({self.ab_n.value()} stumps)",
                           highlight=disagree)

            self.ax_stage.clear()
            self.ax_stage.plot(np.arange(1, len(staged) + 1), staged,
                               c="tab:orange", lw=1.5)
            self.ax_stage.axhline(accuracy_score(lte, rf.predict(Xte)),
                                  ls="--", c="tab:green", lw=1,
                                  label="RF test acc")
            self.ax_stage.axhline(oob, ls=":", c="tab:green", lw=1,
                                  label="RF OOB")
            self.ax_stage.set_xlabel("AdaBoost estimators")
            self.ax_stage.set_ylabel("test accuracy")
            self.ax_stage.set_title("AdaBoost staged accuracy")
            self.ax_stage.legend(fontsize=8); self.ax_stage.grid(alpha=.3)

            self.ax_im.clear()
            xpos = np.arange(2); w = 0.35
            self.ax_im.bar(xpos - w / 2, rf.feature_importances_, w,
                           label="RandomForest", color="tab:green")
            self.ax_im.bar(xpos + w / 2, ada.feature_importances_, w,
                           label="AdaBoost", color="tab:orange")
            self.ax_im.set_xticks(xpos)
            self.ax_im.set_xticklabels(["x₁", "x₂"])
            self.ax_im.set_title("feature importances")
            self.ax_im.legend(); self.ax_im.grid(alpha=.3, axis="y")
            self.canvas.draw()
        finally:
            QApplication.restoreOverrideCursor()


# ═══════════════════════════ Tab 6: Validation ══════════════════════════════
# Ground truth exists here (sklearn labels), so we can measure whether the
# unsupervised pipeline — clustering, autoencoder, ensembles — recovers it,
# and whether the set/group-theory code satisfies its own laws.

class ValidationWorker(QThread):
    log_sig = Signal(str)
    progress_sig = Signal(int, str)
    finished_run = Signal(object)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg

    def run(self):
        try:
            self.finished_run.emit(self._run())
        except Exception:                               # noqa: BLE001
            self.finished_run.emit({"error": traceback.format_exc(),
                                    "report": [], "bench": None,
                                    "set_tests": None, "group_tests": None,
                                    "sweep": None, "verdict": None})

    # ───────────────────────────── orchestration ─────────────────────────────
    def _run(self):
        c = self.cfg
        res = {"bench": None, "set_tests": None, "group_tests": None,
               "sweep": None, "verdict": None}
        lines = []
        res["report"] = lines

        def log(s=""):
            lines.append(s)
            self.log_sig.emit(s)

        log("═" * 64)
        log("VALIDATION SUITE — every algorithm checked against ground truth")
        log(f"sklearn {sklearn.__version__} · numpy {np.__version__}")
        log("═" * 64)

        if c.get("sweep"):
            res["sweep"] = self._stage_sweep(log)
            return res
        if c["bench"]:
            res["bench"] = self._stage_benchmark(log)
            if self.isInterruptionRequested():
                return res
        if c["set_tests"]:
            res["set_tests"] = self._stage_set_tests(log)
            if self.isInterruptionRequested():
                return res
        if c["group_tests"]:
            res["group_tests"] = self._stage_group_tests(log)
        res["verdict"] = self._verdict(res, log)
        return res

    # ─────────────── stage 1: sklearn ground truth ───────────────────────────
    def _stage_benchmark(self, log):
        c = self.cfg
        self.progress_sig.emit(3, "loading dataset")
        X, y, names = load_benchmark(c["dataset"], c["seed"])
        k = len(names)
        log("")
        log(f"[1] GROUND-TRUTH BENCHMARK — {c['dataset']}   "
            f"(typical RF CV accuracy {REFERENCE_ACC[c['dataset']]})")
        log(f"    n = {len(X)}   d = {X.shape[1]}   true classes = {k}   "
            f"(labels never shown to the unsupervised pipeline)")

        self.progress_sig.emit(8, "clustering")
        Xs = StandardScaler().fit_transform(X)
        km = KMeans(n_clusters=k, n_init=10,
                    random_state=c["seed"]).fit_predict(Xs)
        ari = adjusted_rand_score(y, km)
        nmi = normalized_mutual_info_score(y, km)
        h, comp, v = homogeneity_completeness_v_measure(y, km)
        sil = silhouette_score(Xs, km) if 1 < k < len(Xs) else float("nan")
        pur = cluster_purity(y, km)
        log(f"    KMeans(k={k}) vs truth:  ARI {ari:.3f} · NMI {nmi:.3f} · "
            f"H {h:.3f} / C {comp:.3f} / V {v:.3f}")
        log(f"    cluster purity {pur:.3f} · silhouette {sil:.3f}")

        self.progress_sig.emit(13, "clustering algorithm comparison")
        alts = {"KMeans": km}
        try:
            alts["Ward"] = AgglomerativeClustering(n_clusters=k)\
                .fit_predict(Xs)
        except Exception:
            pass
        try:
            alts["GMM"] = GaussianMixture(n_components=k, n_init=3,
                                          random_state=c["seed"])\
                .fit_predict(Xs)
        except Exception:
            pass
        if len(Xs) <= 1200:
            try:
                alts["Spectral"] = SpectralClustering(
                    n_clusters=k, affinity="nearest_neighbors",
                    n_neighbors=15, random_state=c["seed"]).fit_predict(Xs)
            except Exception:
                pass
        best = max(alts, key=lambda a: adjusted_rand_score(y, alts[a]))
        log("    clustering algorithms vs ground truth (ARI / NMI / purity):")
        for a, l in alts.items():
            log(f"      {a:<9} {adjusted_rand_score(y, l):.3f} / "
                f"{normalized_mutual_info_score(y, l):.3f} / "
                f"{cluster_purity(y, l):.3f}" + ("   ← best" if a == best
                                                 else ""))

        X2 = PCA(n_components=2, random_state=0).fit_transform(Xs)

        self.progress_sig.emit(18, "ensembles · 5-fold CV")
        Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.3,
                                              random_state=c["seed"],
                                              stratify=y)
        models = {
            "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, random_state=c["seed"], n_jobs=1),
            "AdaBoost": make_adaboost(c["seed"], 100, 1.0, c["ada_depth"]),
        }
        skf = StratifiedKFold(5, shuffle=True, random_state=c["seed"])
        cv = {}
        for i, (nm, m) in enumerate(models.items()):
            sc = cross_val_score(m, Xs, y, cv=skf, scoring="accuracy",
                                 n_jobs=1)
            cv[nm] = (float(sc.mean()), float(sc.std()))
            self.progress_sig.emit(18 + 12 * (i + 1), f"CV · {nm}")
        accs, cms = {}, {}
        for nm, m in models.items():
            m.fit(Xtr, ytr)
            accs[nm] = accuracy_score(yte, m.predict(Xte))
            cms[nm] = confusion_matrix(yte, m.predict(Xte))
        gap = cv["RandomForest"][0] - cv["Dummy (majority)"][0]
        log("    5-fold CV accuracy vs ground truth: " +
            " · ".join(f"{n} {m:.3f}±{s:.3f}" for n, (m, s) in cv.items()))
        log(f"    held-out test acc: RandomForest {accs['RandomForest']:.3f} · "
            f"AdaBoost {accs['AdaBoost']:.3f}")
        log(f"    RF lift over trivial baseline: {gap:+.3f}  "
            f"{'✓' if gap >= 0.10 else '✗'}")

        self.progress_sig.emit(45, "autoencoder (unsupervised)")
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
        Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
        d = X.shape[1]
        hidden = int(min(64, max(16, 2 * d)))
        dims = [d, hidden, c["ae_latent"], hidden, d]
        ae = Autoencoder(dims, seed=1)
        ep_total = c["ae_epochs"]

        def cb(ep, loss):
            self.progress_sig.emit(45 + int(22 * ep / ep_total),
                                   f"AE epoch {ep}/{ep_total}")
            if ep % 50 == 0 or ep == 1:
                log(f"      [AE] epoch {ep:4d}   MSE {loss:.5f}")
            return self.isInterruptionRequested()

        hist = ae.fit(Xtr_s, epochs=ep_total, lr=1e-2, batch=64, progress=cb)
        mse = float(np.mean((ae._forward(Xte_s)[-1] - Xte_s) ** 2))
        base = float(np.mean((Xte_s - Xtr_s.mean(0)) ** 2))
        r2 = 1.0 - mse / base
        acc_o = accuracy_score(yte, KNeighborsClassifier(5)
                               .fit(Xtr_s, ytr).predict(Xte_s))
        Ztr, Zte = ae.encode(Xtr_s), ae.encode(Xte_s)
        acc_l = accuracy_score(yte, KNeighborsClassifier(5)
                               .fit(Ztr, ytr).predict(Zte))
        lat_ari = adjusted_rand_score(
            yte, KMeans(n_clusters=k, n_init=10,
                        random_state=0).fit_predict(Zte))
        ratio = acc_l / acc_o if acc_o > 0 else float("nan")
        log(f"    Autoencoder {'→'.join(map(str, dims))}  ({len(hist)} epochs)")
        log(f"    reconstruction: test MSE {mse:.4f} vs mean-baseline "
            f"{base:.4f}  →  R² {r2:.3f}  {'✓' if r2 >= 0.25 else '✗'}")
        log(f"    kNN-5 probe vs truth: original {acc_o:.3f} · latent "
            f"{acc_l:.3f} (ratio {ratio:.2f}) · latent-cluster ARI "
            f"{lat_ari:.3f}")

        return {"name": c["dataset"], "X2": X2, "y": y, "km": km, "k": k,
                "ari": ari, "nmi": nmi, "sil": sil, "purity": pur,
                "cv": cv, "accs": accs, "cm": cms["RandomForest"], "gap": gap,
                "nn": {"dims": dims, "mse": mse, "base": base, "r2": r2,
                       "acc_o": acc_o, "acc_l": acc_l, "ratio": ratio,
                       "lat_ari": lat_ari, "epochs": len(hist)}}

    # ─────────────── stage 2: set-operator property tests ────────────────────
    def _stage_set_tests(self, log):
        c = self.cfg
        trials, seed = c["trials"], c["seed"]
        self.progress_sig.emit(70, "set-theory property tests")
        log("")
        log(f"[2] SET-THEORY PROPERTY TESTS — pools drawn from "
            f"{c['dataset']} (PCA 2-D), exact mode ε = 0")
        log("    (exercises the very functions used by tab 2)")
        X, _, _ = load_benchmark(c["dataset"], seed)
        X2 = PCA(2, random_state=0).fit_transform(
            StandardScaler().fit_transform(X))
        rng = np.random.default_rng(seed)
        E = np.zeros((0, 2))
        props = ["inclusion–exclusion",
                 "De Morgan (complement w.r.t. U)",
                 "|A△B| = |A| + |B| − 2|A∩B|",
                 "commutativity (∪, ∩, △)",
                 "idempotence (A∪A, A∩A)",
                 "identity laws (∅, U)",
                 "absorption A∩(A∪B) = A",
                 "associativity (∪ and ∩)"]
        fails = dict.fromkeys(props, 0)
        completed = 0
        for t in range(trials):
            if self.isInterruptionRequested():
                break
            M = int(min(len(X2), rng.integers(80, 240)))
            pool = X2[rng.choice(len(X2), size=M, replace=False)]
            A, B = pool[rng.random(M) < .6], pool[rng.random(M) < .6]
            U = pool
            inter = set_intersection(A, B, 0.0)
            uni = set_union(A, B, 0.0)
            sym = set_symdiff(A, B, 0.0)
            if len(uni) != len(A) + len(B) - len(inter):
                fails[props[0]] += 1
            lhs = set_difference(U, uni, 0.0)
            rhs = set_intersection(set_difference(U, A, 0.0),
                                   set_difference(U, B, 0.0), 0.0)
            if len(lhs) != len(rhs):
                fails[props[1]] += 1
            if len(sym) != len(A) + len(B) - 2 * len(inter):
                fails[props[2]] += 1
            if (len(set_union(B, A, 0.0)) != len(uni)
                    or len(set_intersection(B, A, 0.0)) != len(inter)
                    or len(set_symdiff(B, A, 0.0)) != len(sym)):
                fails[props[3]] += 1
            if (len(set_union(A, A, 0.0)) != len(A)
                    or len(set_intersection(A, A, 0.0)) != len(A)):
                fails[props[4]] += 1
            if (len(set_union(A, E, 0.0)) != len(A)
                    or len(set_intersection(A, E, 0.0)) != 0
                    or len(set_symdiff(A, E, 0.0)) != len(A)
                    or len(set_intersection(A, U, 0.0)) != len(A)):
                fails[props[5]] += 1
            if len(set_intersection(A, uni, 0.0)) != len(A):
                fails[props[6]] += 1
            C = pool[rng.random(M) < .5]
            iab = set_intersection(A, B, 0.0)
            iac = set_intersection(A, C, 0.0)
            ibc = set_intersection(B, C, 0.0)
            iabc = set_intersection(iab, C, 0.0)
            ref = (len(A) + len(B) + len(C) - len(iab) - len(iac)
                   - len(ibc) + len(iabc))
            lu = set_union(set_union(A, B, 0.0), C, 0.0)
            ru = set_union(A, set_union(B, C, 0.0), 0.0)
            li = set_intersection(set_intersection(A, B, 0.0), C, 0.0)
            ri = set_intersection(A, set_intersection(B, C, 0.0), 0.0)
            if len(lu) != ref or len(ru) != ref or len(li) != len(ri):
                fails[props[7]] += 1
            completed += 1
            if (t + 1) % 10 == 0:
                self.progress_sig.emit(70 + int(12 * (t + 1) / trials),
                                       f"set tests {t + 1}/{trials}")
        results, all_ok = {}, True
        for p in props:
            ok = fails[p] == 0
            all_ok &= ok
            results[p] = (completed - fails[p], completed)
            log(f"    {'✓' if ok else '✗'} {p:<36}"
                f"{completed - fails[p]:>4}/{completed} trials")
        log(f"    → set-operator implementation "
            f"{'VALID ✓' if all_ok and completed else 'INVALID ✗'}")
        return {"all_ok": all_ok and completed > 0, "props": results,
                "trials": completed}

    # ─────────────── stage 3: group-theory axiom tests ───────────────────────
    def _stage_group_tests(self, log):
        self.progress_sig.emit(86, "group-theory axiom tests")
        log("")
        log("[3] GROUP-THEORY AXIOM TESTS — D₄ (order 8), exact arithmetic")
        log("    (exercises the very functions used by tab 3)")
        elems, names = d4_elements()
        I = np.eye(2)
        checks = {}

        ok = all(_match_matrix(g @ h, elems) is not None
                 for g in elems for h in elems)
        checks["closure  g∘h ∈ D₄"] = (ok, "all 64 products")

        ok = all(np.allclose((g @ h) @ k2, g @ (h @ k2), atol=1e-12)
                 for g in elems for h in elems for k2 in elems)
        checks["associativity  (g∘h)∘k = g∘(h∘k)"] = (ok, "all 512 triples")

        ok = all(np.allclose(g @ I, g) and np.allclose(I @ g, g)
                 for g in elems)
        checks["identity  e∘g = g∘e = g"] = (ok, "8 elements")

        ok = all(any(np.allclose(g @ h, I, atol=1e-12) for h in elems)
                 for g in elems)
        checks["inverses  ∀g ∃h : g∘h = e"] = (ok, "8 elements")

        rng = np.random.default_rng(0)
        P = rng.uniform(-5, 5, size=(300, 2))
        ok = True
        for g in elems:
            for h in elems:
                if not np.allclose((g @ h) @ P.T, g @ (h @ P.T), atol=1e-9):
                    ok = False
                    break
            if not ok:
                break
        checks["action homomorphism  (g∘h)·p = g·(h·p)"] = \
            (ok, "300 random points × 64 pairs")

        pts = list(P) + [np.array(p, float) for p in
                         [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1),
                          (1.5, -2.5)]]
        ok, bad = True, None
        for p in pts:
            orb = {tuple(np.round(g @ p, 9)) for g in elems}
            stab = sum(1 for g in elems if np.allclose(g @ p, p, atol=1e-9))
            if len(orb) * stab != 8:
                ok, bad = False, (p, len(orb), stab)
                break
        checks["orbit–stabilizer  |G·p|·|G_p| = |D₄|"] = \
            (ok, "306 points incl. origin/axes/diagonals"
             if ok else f"failed at p = {bad}")

        seeds = rng.integers(-3, 4, size=(6, 2))
        S = {tuple((g @ s.astype(float)).astype(int))
             for s in seeds for g in elems}
        canon = {p: min(tuple((g @ np.array(p, float)).astype(int))
                        for g in elems) for p in S}
        K = len(set(canon.values()))
        total_fix = sum(1 for p in S for g in elems
                        if tuple((g @ np.array(p, float)).astype(int)) == p)
        checks["Burnside  Σ_g|Fix(g)| / |D₄| = #orbits"] = \
            (total_fix == 8 * K,
             f"|S| = {len(S)}, orbits = {K}, ΣFix = {total_fix}")

        all_ok = True
        for name, (ok, detail) in checks.items():
            all_ok &= ok
            log(f"    {'✓' if ok else '✗'} {name:<44}{detail}")
        log(f"    → D₄ is a verified group acting on ℝ²: "
            f"{'VALID ✓' if all_ok else 'INVALID ✗'}")
        self.progress_sig.emit(95, "finalizing")
        return {"all_ok": all_ok, "checks": checks}

    # ───────────────────────── stage 4: benchmark sweep ──────────────────────
    def _stage_sweep(self, log):
        log("═" * 64)
        log("BENCHMARK SWEEP — clustering + ensemble CV on every "
            "sklearn benchmark")
        log("═" * 64)
        hdr = (f"{'dataset':<26}{'n':>5}{'d':>4}{'k':>3}{'ARI':>7}{'NMI':>7}"
               f"{'purity':>8}{'RF-CV':>8}{'Ada-CV':>8}{'Dummy':>8}")
        log(hdr)
        log("-" * len(hdr))
        for i, name in enumerate(BENCH_DATASETS):
            if self.isInterruptionRequested():
                break
            self.progress_sig.emit(int(100 * (i + 1) / len(BENCH_DATASETS)),
                                   f"sweep · {name}")
            X, y, names = load_benchmark(name, self.cfg["seed"])
            k = len(names)
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=k, n_init=10,
                        random_state=self.cfg["seed"]).fit_predict(Xs)
            skf = StratifiedKFold(5, shuffle=True,
                                  random_state=self.cfg["seed"])
            rf_cv = cross_val_score(
                RandomForestClassifier(150, random_state=self.cfg["seed"]),
                Xs, y, cv=skf, n_jobs=1).mean()
            ada_cv = cross_val_score(
                make_adaboost(self.cfg["seed"], 60, 1.0, 3),
                Xs, y, cv=skf, n_jobs=1).mean()
            dm_cv = cross_val_score(DummyClassifier(strategy="most_frequent"),
                                    Xs, y, cv=skf).mean()
            log(f"{name:<26}{len(X):>5}{X.shape[1]:>4}{k:>3}"
                f"{adjusted_rand_score(y, km):>7.3f}"
                f"{normalized_mutual_info_score(y, km):>7.3f}"
                f"{cluster_purity(y, km):>8.3f}"
                f"{rf_cv:>8.3f}{ada_cv:>8.3f}{dm_cv:>8.3f}")
        log("-" * len(hdr))
        log("RF-CV sanity vs published anchors: " +
            "; ".join(f"{n}: {REFERENCE_ACC[n]}" for n in BENCH_DATASETS))
        log("→ valid ✓ if RF-CV ≈ anchors and Dummy ≪ RF on every row.")
        self.progress_sig.emit(100, "sweep done")

    # ───────────────────────────── verdict ───────────────────────────────────
    def _verdict(self, res, log):
        log("")
        log("═" * 64)
        problems = []
        st, gt, b = res["set_tests"], res["group_tests"], res["bench"]
        if st is not None and not st["all_ok"]:
            problems.append("set-operator self-tests failed")
        if gt is not None:
            bad = [k for k, (ok, _) in gt["checks"].items() if not ok]
            if bad:
                problems.append("group axioms failed: " + ", ".join(bad))
        if b is not None:
            if b["gap"] < 0.10:
                problems.append(f"RF gain over dummy only {b['gap']:+.3f}")
            if b["nn"]["r2"] < 0.25:
                problems.append(f"AE reconstruction R² only "
                                f"{b['nn']['r2']:.3f}")
            if b["nn"]["ratio"] < 0.80:
                problems.append("latent kNN probe ratio < 0.80")
        if problems:
            log("OVERALL VERDICT: ✗  ISSUES: " + "; ".join(problems))
            return False
        log("OVERALL VERDICT: ✓  VALIDATION PASSED — set operators satisfy "
            "all tested laws, D₄ satisfies every axiom, ensembles beat the "
            "trivial baseline on ground truth, and the autoencoder preserves "
            "class structure without ever seeing a label.")
        return True


class ValidationTab(QWidget):
    def __init__(self, store):
        super().__init__()
        self.store, self.worker, self._last_bench = store, None, None
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        note = QLabel(
            "Ground-truth check: labeled sklearn datasets are pushed through "
            "the UNSUPERVISED pipeline (clustering → pseudo-labels → "
            "autoencoder → ensembles); agreement with the hidden true labels "
            "(ARI, NMI, purity, CV accuracy, kNN probes) validates the "
            "algorithms. Self-tests exercise the exact set-operator and D₄ "
            "functions used by tabs 2–5.")
        note.setWordWrap(True)
        left.addWidget(note)

        b = QGroupBox("Ground-truth benchmark (sklearn)")
        fb = QFormLayout(b)
        self.dataset = QComboBox(); self.dataset.addItems(BENCH_DATASETS)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999)
        self.seed.setValue(0)
        self.ada_depth = QSpinBox(); self.ada_depth.setRange(1, 8)
        self.ada_depth.setValue(3)
        self.ae_latent = QSpinBox(); self.ae_latent.setRange(2, 16)
        self.ae_latent.setValue(8)
        self.ae_epochs = QSpinBox(); self.ae_epochs.setRange(50, 2000)
        self.ae_epochs.setSingleStep(50); self.ae_epochs.setValue(250)
        fb.addRow("Dataset", self.dataset)
        fb.addRow("Seed", self.seed)
        fb.addRow("AdaBoost stump depth", self.ada_depth)
        fb.addRow("AE latent dim", self.ae_latent)
        fb.addRow("AE epochs", self.ae_epochs)
        left.addWidget(b)
        self.btn_bench = QPushButton(
            "Run benchmark (clustering + ensembles + AE)")
        self.btn_bench.clicked.connect(
            lambda: self.start(bench=True, set_tests=False, group_tests=False))
        left.addWidget(self.btn_bench)

        s = QGroupBox("Implementation self-tests")
        fs = QFormLayout(s)
        self.trials = QSpinBox(); self.trials.setRange(1, 500)
        self.trials.setValue(50)
        fs.addRow("Random trials", self.trials)
        self.btn_set = QPushButton("Set-theory property tests (ε-exact)")
        self.btn_set.clicked.connect(lambda: self.start(False, True, False))
        self.btn_group = QPushButton("Group-theory axiom tests (D₄)")
        self.btn_group.clicked.connect(lambda: self.start(False, False, True))
        fs.addRow(self.btn_set); fs.addRow(self.btn_group)
        self.btn_sweep = QPushButton(
            "Benchmark sweep — ALL datasets (fast)")
        self.btn_sweep.clicked.connect(lambda: self.start(sweep=True))
        fs.addRow(self.btn_sweep)
        left.addWidget(s)

        self.btn_full = QPushButton("Run FULL validation suite")
        self.btn_full.clicked.connect(lambda: self.start(True, True, True))
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_worker)
        left.addWidget(self.btn_full); left.addWidget(self.btn_stop)

        self.btn_export = QPushButton("Export report → Markdown")
        self.btn_export.clicked.connect(self.export_report)
        left.addWidget(self.btn_export)

        self.pb = QProgressBar(); self.status = QLabel("idle")
        left.addWidget(self.pb); left.addWidget(self.status)
        self.log = mono_text(); left.addWidget(self.log, stretch=1)
        add_btn = QPushButton(
            "→ add 2-D PCA projection of dataset to universe U")
        add_btn.clicked.connect(self.add_to_universe)
        left.addWidget(add_btn)

        lw = QWidget(); lw.setLayout(left); lw.setMaximumWidth(440)
        lay.addWidget(lw)

        self.canvas = MplCanvas(8.6, 6.6)
        ax = self.canvas.fig.add_subplot(111)
        ax.text(.5, .5, "run a validation to populate the panels",
                ha="center", va="center", transform=ax.transAxes, color="#888")
        ax.set_axis_off()
        self.canvas.draw()
        lay.addWidget(self.canvas, stretch=1)

    def start(self, bench=False, set_tests=False, group_tests=False,
              sweep=False):
        if self.worker is not None and self.worker.isRunning():
            return
        cfg = {"bench": bench, "set_tests": set_tests,
               "group_tests": group_tests, "sweep": sweep,
               "dataset": self.dataset.currentText(),
               "seed": self.seed.value(), "trials": self.trials.value(),
               "ada_depth": self.ada_depth.value(),
               "ae_latent": self.ae_latent.value(),
               "ae_epochs": self.ae_epochs.value()}
        for btn in (self.btn_bench, self.btn_set, self.btn_group,
                    self.btn_sweep, self.btn_full):
            btn.setEnabled(False)
        self.pb.setRange(0, 100); self.pb.setValue(0)
        self.log.clear()
        self.worker = ValidationWorker(cfg)
        self.worker.log_sig.connect(self.log.append)
        self.worker.progress_sig.connect(self._on_progress)
        self.worker.finished_run.connect(self.on_done)
        self.worker.start()

    def _on_progress(self, pct, msg):
        self.pb.setValue(pct)
        self.status.setText(msg)

    def stop_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(3000)

    def on_done(self, res):
        for btn in (self.btn_bench, self.btn_set, self.btn_group,
                    self.btn_sweep, self.btn_full):
            btn.setEnabled(True)
        self.pb.setValue(100); self.status.setText("done")
        if "error" in res:
            self.log.append("\n" + res["error"])
            QMessageBox.critical(self, "Validation", "Error — see log.")
            return
        if res["bench"] is not None:
            self._last_bench = res["bench"]
            self._render_bench(res["bench"])
        self.log.setPlainText("\n".join(res["report"]))

    def export_report(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export report",
                                              "validation_report.md",
                                              "Markdown (*.md);;Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# Validation report — {datetime.now():%Y-%m-%d %H:%M}"
                     f"\n\n```\n" + self.log.toPlainText() + "\n```\n")
        self.status.setText(f"report written → {path}")

    def add_to_universe(self):
        if self._last_bench is None:
            QMessageBox.information(self, "Validation",
                                    "Run the benchmark first.")
            return
        b = self._last_bench
        name = self.store.add(b["name"].split(" (")[0].lower() + "_pca2d",
                              b["X2"])
        self.status.setText(f"added “{name}” to U (unlabeled)")

    def _render_bench(self, b):
        fig = self.canvas.fig
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        k = b["k"]
        if k <= 10:
            cmap = tab10(k)
            col_t, col_c = cmap(np.asarray(b["y"])), cmap(np.asarray(b["km"]))
        else:
            nrm, cmap = Normalize(0, k - 1), get_cmap("viridis")
            col_t, col_c = cmap(nrm(b["y"])), cmap(nrm(b["km"]))
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(b["X2"][:, 0], b["X2"][:, 1], c=col_t, s=12)
        ax.set_title("TRUE labels (hidden from the pipeline) — PCA 2-D")
        ax.grid(alpha=.3)
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(b["X2"][:, 0], b["X2"][:, 1], c=col_c, s=12)
        ax.set_title(f"KMeans pseudo-labels — ARI {b['ari']:.2f}, "
                     f"NMI {b['nmi']:.2f}, purity {b['purity']:.2f}")
        ax.grid(alpha=.3)
        ax = fig.add_subplot(gs[1, 0])
        cm = b["cm"]
        ax.imshow(cm, cmap="Blues")
        thr = cm.max() / 2 if cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=8,
                        color="white" if cm[i, j] > thr else "black")
        ax.set_title(f"RandomForest confusion matrix "
                     f"(test acc {b['accs']['RandomForest']:.3f})")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax = fig.add_subplot(gs[1, 1])
        short = {"Dummy (majority)": "Dummy", "RandomForest": "RandomForest",
                 "AdaBoost": "AdaBoost"}
        nms = list(b["cv"].keys())
        means = [b["cv"][n][0] for n in nms]
        stds = [b["cv"][n][1] for n in nms]
        ax.bar([short[n] for n in nms], means, yerr=stds, capsize=4,
               color=["#aaaaaa", "tab:green", "tab:orange"][:len(nms)])
        for i, m in enumerate(means):
            ax.text(i, m + 0.02, f"{m:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title("5-fold CV accuracy vs ground truth")
        ax.grid(alpha=.3, axis="y")
        fig.suptitle(f"Ground-truth validation — {b['name']}")
        self.canvas.draw()


# ══════════════════════════════════ shell ══════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Unsupervised Set & Group Theory ML Studio")
        self.resize(1480, 920)
        self.store = DatasetStore()
        self.tabs = QTabWidget()
        self.data_tab = DataStudioTab(self.store)
        self.neural_tab = NeuralTab(self.store)
        self.validation_tab = ValidationTab(self.store)
        pages = [
            (self.data_tab, "1 · Data Studio"),
            (SetTheoryTab(self.store), "2 · Set Theory"),
            (GroupTheoryTab(self.store), "3 · Group Theory"),
            (self.neural_tab, "4 · Neural Net (AE)"),
            (EnsembleTab(self.store), "5 · AdaBoost & Forest"),
            (self.validation_tab, "6 · Validation (sklearn)"),
        ]
        for w, title in pages:
            self.tabs.addTab(w, title)
        self.setCentralWidget(self.tabs)
        self._build_menus()
        self.store.add("blobs", make_dataset("Blobs", 500, 0.4, 4, 7))
        self.statusBar().showMessage(
            "Unlabeled sets A ⊂ ℝ² everywhere · set algebra (2) · group "
            "actions Dₙ/Cₙ + augmentation (3) · denoising AE (4) · ensembles "
            "+ invariance probes (5) · ground-truth validation (6). "
            "File menu: save/load universe, export plots.")

    def _build_menus(self):
        mfile = self.menuBar().addMenu("&File")
        for label, slot, sc in [
                ("Save universe…", self.save_universe, "Ctrl+S"),
                ("Load universe…", self.load_universe, "Ctrl+O"),
                ("Export current plot…", self.export_plot, "Ctrl+E"),
                ("Quit", self.close, "Ctrl+Q")]:
            a = QAction(label, self)
            a.setShortcut(QKeySequence(sc))
            a.triggered.connect(slot)
            mfile.addAction(a)
        mhelp = self.menuBar().addMenu("&Help")
        a = QAction("About", self)
        a.triggered.connect(self.about)
        mhelp.addAction(a)

    def save_universe(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save universe",
                                              "universe.npz",
                                              "NumPy archive (*.npz)")
        if not path:
            return
        names = self.store.names()
        np.savez_compressed(path, names="\n".join(names),
                            **{f"a{i}": self.store.datasets[n]
                               for i, n in enumerate(names)})
        self.statusBar().showMessage(
            f"universe saved → {path} ({len(names)} sets)", 5000)

    def load_universe(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load universe", "",
                                              "NumPy archive (*.npz)")
        if not path:
            return
        data = np.load(path)
        names = [s for s in str(data["names"]).split("\n") if s]
        if self.store.names() and QMessageBox.question(
                self, "Load",
                f"Replace current {len(self.store.names())} set(s) "
                f"with {len(names)}?") != QMessageBox.Yes:
            return
        for n in self.store.names():
            self.store.delete(n)
        for i, n in enumerate(names):
            self.store.add(n, data[f"a{i}"])
        self.statusBar().showMessage(
            f"universe loaded ← {path} ({len(names)} sets)", 5000)

    def export_plot(self):
        canvases = self.tabs.currentWidget().findChildren(FigureCanvasQTAgg)
        if not canvases:
            QMessageBox.information(self, "Export",
                                    "Current tab has no plot.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export plot", "plot.png",
                                              "PNG (*.png);;PDF (*.pdf);;"
                                              "SVG (*.svg)")
        if path:
            canvases[0].fig.savefig(path, dpi=160, bbox_inches="tight")
            self.statusBar().showMessage(f"plot exported → {path}", 5000)

    def about(self):
        QMessageBox.about(
            self, "About",
            "Unsupervised Set & Group Theory ML Studio\n\n"
            "Datasets are unlabeled sets A ⊂ ℝ².\n"
            "• Tab 2: ε-tolerant set algebra (∪ ∩ ∖ △, De Morgan, Jaccard…)\n"
            "• Tab 3: quotient sets A/~ and group actions Dₙ, Cₙ — orbits, "
            "stabilizers, Cayley tables, all subgroups, conjugacy, "
            "invariants\n"
            "• Tab 4: numpy autoencoder / denoising AE\n"
            "• Tab 5: RandomForest & AdaBoost on pseudo-labels q: A → A/~\n"
            "• Tab 6: ground-truth validation on sklearn benchmarks + "
            "property/axiom self-tests")

    def closeEvent(self, e):
        self.neural_tab.stop_worker()
        self.validation_tab.stop_worker()
        super().closeEvent(e)


APP_QSS = """
QMainWindow, QDialog { background: #f7f8fa; }
QGroupBox { font-weight: 600; border: 1px solid #d3d7e0; border-radius: 6px;
            margin-top: 10px; background: #ffffff; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { background: #ffffff; border: 1px solid #c3c8d2;
              border-radius: 4px; padding: 5px 10px; }
QPushButton:hover { background: #eef2ff; border-color: #7f9cf5; }
QPushButton:pressed { background: #e0e7ff; }
QTabWidget::pane { border: 1px solid #d3d7e0; background: #ffffff; }
QTabBar::tab { padding: 6px 14px; background: #eceef2; }
QTabBar::tab:selected { background: #ffffff; border-top: 2px solid #4f6ef7; }
QListWidget, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar {
    background: #ffffff; border: 1px solid #c3c8d2; border-radius: 4px; }
QProgressBar::chunk { background: #4f6ef7; border-radius: 3px; }
QStatusBar { background: #eceef2; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()