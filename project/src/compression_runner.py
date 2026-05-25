import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Any, Dict, Tuple
import os
from torchvision import models
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau


class CompressionHead(nn.Module):
    # Simple MLP compression head (Non-linear Probing)
    # with ReLU activation function often populirsed by methods such as simclr
    #
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class CompressionRunner:
    def __init__(
            self,
            config: Dict[str, Any],
            benchmark_runner: Any,
            model: Any,
            train_embeddings: np.ndarray,
            train_labels: np.ndarray,
            eval_embeddings: np.ndarray,
            eval_labels: np.ndarray,
            effective_rank: float,
            num_classes: int,
            device : torch.device,
    ):
        self.config = config
        self.runner = benchmark_runner
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.eval_embeddings = eval_embeddings
        self.eval_labels = eval_labels
        self.effective_rank = effective_rank
        self.num_classes = num_classes

        # model
        self.base_model = model

        # device for self distilation
        self.step_size = self.config.get("compression_step_size", 50)
        self.device = device

        # Set up figure directory
        self.run_name = self.config.get("name", "benchmark_run")
        self.figures_dir = Path(self.config.get("figures_dir", "project/models_out/figures")) / self.run_name
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Set up report directory
        self.report_dir = Path(self.config.get("report_dir", "project/models_out/reports")) / self.run_name
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Internal log container
        self.report_logs = []

    def _log_and_print(self, msg: str):
        """Prints to console and stores message for the final text report."""
        print(msg)
        self.report_logs.append(msg)

    def run_pca_compression(self):
        self._log_and_print(f"\n========================================================")
        self._log_and_print(
            f" PCA Compression (Maximum dimentions: {self.train_embeddings.shape[1]}, Effective Rank: {self.effective_rank:.1f})")
        self._log_and_print(f"========================================================\n")

        max_dim = self.train_embeddings.shape[1]
        start_dim = min(max_dim, int(self.effective_rank) + 100)

        # list: start_dim  step: lower_checkpoint - 50, step - self.step_size
        dimensions = list(range(start_dim, 49, -self.step_size))
        if dimensions[-1] != 50 and 50 not in dimensions:
            dimensions.append(50)  # Upewniamy się, że zbadamy wymiar 50

        results_dim = []
        results_knn1 = []
        results_knn5 = []
        results_linear = []

        for dim in dimensions:
            self._log_and_print(f"[{dim}D] Space reduction...")
            pca = PCA(n_components=dim, random_state=42)

            #  PCA on train set
            train_comp = pca.fit_transform(self.train_embeddings)
            # transform eval
            eval_comp = pca.transform(self.eval_embeddings)

            # eval using benchmarker functions
            knn_res = self.runner.run_knn_eval(train_comp, self.train_labels, eval_comp, self.eval_labels)
            lin_res = self.runner.run_linear_eval(train_comp, self.train_labels, eval_comp, self.eval_labels)

            acc1 = knn_res["acc1"]
            acc5 = knn_res["acc5"]
            lin_acc = lin_res["accuracy"] * 100  # transfer to percentage

            self._log_and_print(f"   -> Results: kNN@1: {acc1:.2f}% | kNN@5: {acc5:.2f}% | Linear: {lin_acc:.2f}%\n")

            results_dim.append(dim)
            results_knn1.append(acc1)
            results_knn5.append(acc5)
            results_linear.append(lin_acc)

        self._plot_pca_degradation(results_dim, results_knn1, results_knn5, results_linear)

    def _plot_pca_degradation(self, dims, knn1, knn5, linear):
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")

        plt.plot(dims, knn1, marker='o', label='k-NN Acc@1', linewidth=2)
        plt.plot(dims, knn5, marker='s', label='k-NN Acc@5', linewidth=2)
        plt.plot(dims, linear, marker='^', label='Linear Acc', linewidth=2)

        # Odwracamy oś X, żeby wymiary malały od lewej do prawej
        plt.gca().invert_xaxis()

        # Dodajemy pionową linię wskazującą Effective Rank
        plt.axvline(x=self.effective_rank, color='red', linestyle='--',
                    label=f'Effective Rank ({self.effective_rank:.1f})')

        method_name = self.config.get("method", "Model")
        dataset_name = self.config.get("dataset", "Dataset")

        plt.title(f"Degradacja celności przy kompresji PCA - {method_name} na {dataset_name}", fontsize=14)
        plt.xlabel("Liczba komponentów PCA", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()

        filename = f"{method_name}_{dataset_name}_pca_compression.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._log_and_print(f"[Visualisation] Degradation plot PCA saved as: {filename}\n")

    def run_mlp_projector_compression(self):
        self._log_and_print(f"========================================================")
        self._log_and_print("--- (MLP Non-linear Probing) ---")
        self._log_and_print(f"========================================================\n")

        # check popular 3 hidden_dim
        hidden_dims = [512, 256, 128]


        # prepare for training eval split
        train_loader , eval_loader = self.prepare_dataset(2048)
        in_dim = self.train_embeddings.shape[1]

        for h_dim in hidden_dims:
            self._log_and_print(f"[{in_dim} -> {h_dim}D] Training non linear projection...")
            model = CompressionHead(in_dim=in_dim, hidden_dim=h_dim, num_classes=self.num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            epochs = 100

            model.train()
            for epoch in range(epochs):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

            # eval on eval set
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in eval_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = model(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            acc = 100.0 * correct / total
            self._log_and_print(f"   -> Result of non linear compression (Hidden dim: {h_dim}): Top Acc@1 = {acc:.2f}%\n")

        self._save_report("MLP_Projection")

    def prepare_dataset(self, batch_size : int) -> Tuple[DataLoader, DataLoader]:
        # prepare data loaders from RAM stored embeddings
        train_tensor_x = torch.from_numpy(self.train_embeddings).float()
        train_tensor_y = torch.from_numpy(self.train_labels).long()
        eval_tensor_x = torch.from_numpy(self.eval_embeddings).float()
        eval_tensor_y = torch.from_numpy(self.eval_labels).long()

        train_ds = TensorDataset(train_tensor_x, train_tensor_y)
        eval_ds = TensorDataset(eval_tensor_x, eval_tensor_y)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

        return train_loader, eval_loader

    def _save_report(self, name):
        # save all the logs
        filename = f"{self.run_name}_compression_report_{name}.txt"
        filepath = self.report_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_logs))

        print(f"\n[Raport] Final report saved sucessfully as -->  {filepath}")

    # main function for model distilation
    def self_distilation(self):
        self._log_and_print(f"\n[Final compression test] Self-distillation: SSL ResNet-50 -> ResNet-18...")

        #  Dataloaders with pics from BenchmarkRunner
        train_loader = self.runner.create_train_loader()
        eval_loader = self.runner.create_eval_loader()

        # init student and  teacher
        teacher, student_wrapper = self.setup_models()

        # train teachers head (SSL Model)
        self._log_and_print("-> Linear Probing teachers network...")
        teacher = self.train_teacher(teacher, train_loader, eval_loader, epochs=15, tag="Teacher")

        # model distilation
        self._log_and_print("-> Distilate teacher knowladge to smaller cnn...")
        student_model = self.train_student(teacher, student_wrapper, train_loader, eval_loader, epochs=30)

        self._log_and_print("\n-> Compression efficency evaluation...")

        t_params = self.count_params(teacher)
        s_params = self.count_params(student_model)

        # CIFAR -> 32x32,  ImageNet -> 224x224 STL -> 96x96
        img_size = 224 if "imagenet" in self.config.get("dataset", "") else 32

        t_latency = self.measure_latency(teacher, input_size=(1, 3, img_size, img_size), device=self.device)
        s_latency = self.measure_latency(student_model, input_size=(1, 3, img_size, img_size), device=self.device)

        self._log_and_print(f"   Params comparison:")
        self._log_and_print(f"   - Teacher (ResNet-50): {t_params:,} params")
        self._log_and_print(f"   - Student (ResNet-18): {s_params:,} params")
        self._log_and_print(f"   -> Reduced dimensionality by: {100 * (1 - s_params / t_params):.2f}%")

        self._log_and_print(f"   Time difference (Input {img_size}x{img_size}):")
        self._log_and_print(f"   - Teacher Latency: {t_latency:.2f} ms")
        self._log_and_print(f"   - Student Latency: {s_latency:.2f} ms")
        self._log_and_print(f"   -> PSpeedup: {t_latency / s_latency:.2f}x\n")

        # save report
        self._save_report("Distilation")

    def setup_models(self):
        # teacher: fetch ready, benchmarkrunner ssl model
        # resnet50 without head
        #teacher = self.runner.feature_model

        #if teacher is None:
        #    teacher = self.runner.feature_model
        teacher = getattr(self, 'base_model', None)

        if teacher is None:
            teacher = getattr(self.runner, 'feature_model', None)

        if teacher is None:
            raise ValueError("[ERROR] Nie udało się załadować modelu Nauczyciela! Obie zmienne są puste.")


        # if teacher doesn't have  fc att, add it :
        if not hasattr(teacher, 'fc') or teacher.fc is None or isinstance(teacher.fc, nn.Identity):
            teacher.fc = nn.Linear(2048, self.num_classes)

        teacher = teacher.to(self.device)

        # student: empty ResNet18
        student = models.resnet18(weights=None)
        student.fc = nn.Linear(512, self.num_classes)
        student = student.to(self.device)

        # FeatureProjector
        student_channels = [64, 128, 256, 512]
        teacher_channels = [256, 512, 1024, 2048]

        proj_layers = [
            FeatureProjector(in_c, out_c).to(self.device)
            for in_c, out_c in zip(student_channels, teacher_channels)
        ]
        student_wrapper = StudentWrapper(student, proj_layers).to(self.device)

        return teacher, student_wrapper

    def extract_teacher_features(self, model, x, layers=[1, 2, 3, 4]):
        """
        Extract teacher logits and intermediate features
        """

        # collect intermediate features from ResNet blocks
        features = []

        backbone = getattr(model, 'backbone', model)

        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        for i, block in enumerate([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]):
            x = block(x)
            if (i + 1) in layers:
                features.append(x)

        # pool the final feature map and compute logits
        pooled = F.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        flat = torch.flatten(pooled, 1)            # [B, C]
        logits = model.fc(flat)                    # [B, 10]
        return logits, features

    def count_params(self, model, only_trainable=False):
        """
        Function to count trainable parameters
        """

        if only_trainable:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    def measure_latency(self, model, input_size=(1, 3, 32, 32), device='cuda', repetitions=50):
        """
        Function to measure average inference latency over multiple runs
        """

        model.eval()
        inputs = torch.randn(input_size).to(device)
        with torch.no_grad():
            # Warm-up
            for _ in range(10):
                _ = model(inputs)
            # Measure
            times = []
            for _ in range(repetitions):
                start = time.time()
                _ = model(inputs)
                end = time.time()
                times.append(end - start)
        return (sum(times) / repetitions) * 1000  # ms

    def evaluate_accuracy(self, model, dataloader):
        """
        Evaluate accuracy given model and loader
        """

        model.eval()
        model.to(self.device)
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                # if solo-learn model returns a dict, extract features and let them trhough head
                if isinstance(outputs, dict):
                    feats = outputs["feats"]
                    if feats.ndim > 2:
                        feats = torch.flatten(feats, start_dim=1)
                    outputs = model.fc(feats)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        return accuracy



    # this fine tuning required only when the base network was pretrained on ImageNet1K and its output layer needs to match
    # to the final dataset
    def train_teacher(self, teacher, train_loader, eval_loader, epochs, tag, lr=1e-3, save_path="./project/models_out/checkpoints/mine/mineteacher.pth"):
        if os.path.exists(save_path):
            self._log_and_print(f"[{tag}] Model already trained. Loading from {save_path}")
            teacher.load_state_dict(torch.load(save_path))
            return teacher

        # freeze the backbone, train only head to keep the ssl features
        for param in teacher.parameters():
            param.requires_grad = False
        for param in teacher.fc.parameters():
            param.requires_grad = True

        # optimizer only on na fc layer!
        optimizer = torch.optim.Adam(teacher.fc.parameters(), lr=lr)

        for epoch in range(epochs):
            teacher.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # end logits logits
                #logits = teacher(inputs)  # teacher forward
                logits, _ = self.extract_teacher_features(teacher, inputs)

                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accuracy = self.evaluate_accuracy(teacher, eval_loader)
            self._log_and_print(f"   [{tag}] Epoch {epoch + 1}: Val Acc = {accuracy * 100:.2f}%")

        if save_path:
            torch.save(teacher.state_dict(), save_path)

        return teacher

    def distillation_loss(self, student_logits, teacher_logits, targets, T=5.0, alpha=0.7):
        """
        Combine soft and hard targets using KL divergence and cross-entropy
        T = temperature, alpha = weighting between soft and hard losses
        """

        # soft target loss (teacher softmax vs student softmax)
        soft_targets = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        # hard label loss
        hard_loss = F.cross_entropy(student_logits, targets)
        return alpha * soft_targets + (1 - alpha) * hard_loss

    def student_training_step(self, inputs, labels, teacher, student_wrapper, optimizer, device):
        """
        Perform a single training step for the student model using knowledge distillation.
        """

        inputs, labels = inputs.to(device), labels.to(device)

        # extract teacher logits and intermediate features
        with torch.no_grad():
            teacher_logits, teacher_feats = self.extract_teacher_features(teacher, inputs)

        # extract student logits and intermediate features
        student_logits, student_feats = student_wrapper(inputs)
        projected_feats = student_wrapper.project_features(student_feats, [t.shape for t in teacher_feats])

        # calculate loss from features difference
        feat_loss = sum(F.mse_loss(p, t.detach()) for p, t in zip(projected_feats, teacher_feats))

        # calculate loss from output distribution, and include feature loss
        loss = self.distillation_loss(student_logits, teacher_logits, labels) + 0.1 * feat_loss

        # optimize with loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train_student(self, teacher, student_wrapper, dataloader,val_loader, epochs, save_path="./project/models_out/checkpoints/mine/student_distilled.pth"):
        """
        Trains a student model using knowledge distillation from a teacher model.
        """

        # setup optimizer
        optimizer = torch.optim.Adam(student_wrapper.parameters(), lr=1e-3)

        # train the student using the teacher's output as soft targets
        teacher.eval()

        best_val_acc = 0.0

        # reduce LR if validation loss doesn't improve for 3 epochs
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        # for plots
        history_loss = []
        history_acc = []

        for epoch in range(epochs):
            student_wrapper.train()
            running_loss = 0
            for inputs, labels in dataloader:
                loss = self.student_training_step(inputs, labels, teacher, student_wrapper, optimizer, self.device)
                running_loss += loss

            avg_loss = running_loss / len(dataloader)
            val_acc = self.evaluate_accuracy(student_wrapper.model, val_loader)

            # add it to the tables
            history_loss.append(avg_loss)
            history_acc.append(val_acc * 100)
            print(
                f"[(Training student)\tEpoch {epoch + 1}] Loss = {running_loss / len(dataloader):.4f} | Val Acc = {val_acc * 100:.2f}%")
            scheduler.step(avg_loss)

            # save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(student_wrapper.state_dict(), save_path)
                print("New best model saved.")

        # load best checkpoint
        student_wrapper.load_state_dict(torch.load(save_path))
        student = student_wrapper.model

        self._plot_distillation_history(history_loss, history_acc)

        return student

    def _plot_distillation_history(self, losses, accuracies):
        plt.figure(figsize=(12, 5))
        sns.set_style("whitegrid")

        # plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(losses, label='Training Loss (Total)', color='red', linewidth=2)
        plt.title('Krzywa funkcji straty (Distillation)')
        plt.xlabel('Epoka')
        plt.ylabel('Loss')
        plt.legend()

        # plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, label='Validation Accuracy', color='blue', linewidth=2)
        plt.title('Celność klasyfikacji Ucznia')
        plt.xlabel('Epoka')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        filename = f"{self.config.get('method', 'model')}_{self.config.get('dataset', 'dataset')}_distillation_history.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        self._log_and_print(f"[Visualisation] Distillation history plot saved as: {filename}")


class FeatureProjector(nn.Module):

    # Feature projector for matching student -> teacher feature shapes


    def __init__(self, in_channels, out_channels):
        super().__init__()

        # define a 1x1 convolutional layer to project feature maps
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, target_shape):
        # check if the spatial dimensions of the input match the target shape
        if x.shape[2:] != target_shape[2:]:
            # adjust spatial dimensions using adaptive average pooling
            x = F.adaptive_avg_pool2d(x, output_size=target_shape[2:])

        # apply the projection layer to transform feature maps
        return self.proj(x)


class StudentWrapper(nn.Module):
    #  Wrapper class for the student model with projection layers

    def __init__(self, student_model, proj_layers):
        super().__init__()

        # store student model
        self.model = student_model

        # store projection layers for feature alignment
        self.projections = nn.ModuleList(proj_layers)

    def forward(self, x):
        # collect intermediate features from ResNet blocks
        features = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        for i, block in enumerate([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]):
            # pass through ResNet blocks
            x = block(x)

            # append features from each block
            features.append(x)

            # pool the final feature map and compute logits
        pooled = F.adaptive_avg_pool2d(x, (1, 1))
        flat = torch.flatten(pooled, 1)
        logits = self.model.fc(flat)

        return logits, features

    def project_features(self, features, target_shapes):
        # Project student features to match the shapes of teacher features.

        return [
            proj(s_feat, t_shape)
            for s_feat, t_shape, proj in zip(features, target_shapes, self.projections)
        ]



