import torch
import numpy as np
from tqdm import tqdm
import os
import copy

def train_action_classifier(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    save_path='best_action_model.pth',
    use_mixup=False,
    mixup_alpha=0.2
):
    model = model.to(device)

    history = {
        'loss_train': [],
        'acc_train': [],
        'loss_val': [],
        'acc_val': []
    }

    best_acc = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_mixup and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(images.size(0)).to(device)

                mixed_images = lam * images + (1 - lam) * images[index]
                output = model(mixed_images)

                y_a, y_b = labels, labels[index]
                loss = lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)

            else:
                output = model(images)
                loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        history['loss_train'].append(avg_train_loss)
        history['acc_train'].append(avg_train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = criterion(output, labels)

                val_loss += loss.item()
                _, predicted = torch.max(output, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        history['loss_val'].append(avg_val_loss)
        history['acc_val'].append(avg_val_acc)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_acc)
            else:
                scheduler.step()

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            best_state_dict = copy.deepcopy(model.state_dict())

            if save_path is not None:
                torch.save(best_state_dict, save_path)

            saved_msg = f"-> Best model saved (val_acc={best_acc:.4f})"
        else:
            saved_msg = ""

        print(
            f'Epoch [{epoch+1}/{epochs}] '
            f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} {saved_msg}'
        )
        print('-' * 80)

    model.load_state_dict(best_state_dict)

    return model, history, best_acc
