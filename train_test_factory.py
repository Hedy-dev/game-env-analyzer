from torch.utils.data import Dataset, ConcatDataset, Subset
from sklearn.model_selection import train_test_split



class TransformWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

        for attr in ['classes', 'class_to_id', 'id_to_class']:
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        image = image.permute(1, 2, 0).cpu().numpy()

        image = self.transform(image=image)['image']
        return image, label



class RemapDataset(Dataset):
    def __init__(self, dataset, global_class_to_id):
        self.dataset = dataset
        self.global_class_to_id = global_class_to_id

        self.classes = list(global_class_to_id.keys())
        self.class_to_id = global_class_to_id
        self.id_to_class = {v: k for k, v in global_class_to_id.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, local_label = self.dataset[idx]
        class_name = self.dataset.id_to_class[local_label]
        global_label = self.global_class_to_id[class_name]
        return image, global_label



def build_train_val_datasets(
    datasets:list[Dataset],
    val_size:float=0.2,
    random_state:int=42
):
    
    all_classes = set()
    for ds in datasets:
        all_classes.update(ds.classes)

    all_classes = sorted(all_classes)
    global_class_to_id = {c: i for i, c in enumerate(all_classes)}

    remapped_datasets = [
        RemapDataset(ds, global_class_to_id)
        for ds in datasets
    ]

    full_dataset = ConcatDataset(remapped_datasets)

    labels = []
    for ds in remapped_datasets:
        for i in range(len(ds)):
            _, lbl = ds[i]
            labels.append(lbl)

    indices = list(range(len(full_dataset)))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=labels,
        random_state=random_state
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    train_dataset.classes = all_classes
    val_dataset.classes = all_classes
    train_dataset.class_to_id = global_class_to_id
    val_dataset.class_to_id = global_class_to_id
    train_dataset.id_to_class = {v: k for k, v in global_class_to_id.items()}
    val_dataset.id_to_class = {v: k for k, v in global_class_to_id.items()}

    return train_dataset, val_dataset
