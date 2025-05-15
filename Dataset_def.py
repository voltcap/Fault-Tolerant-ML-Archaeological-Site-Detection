class Mesopotamia(Dataset):
    def __init__(self, root_dir, transform=None, is_validation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['no', 'partial', 'yes']
        self.images = []

        if is_validation:
            print("Loading validation data...")
            for img_name in sorted(os.listdir(root_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_lower = img_name.lower()
                    label = 0  
                    if 'partial' in img_lower:
                        label = 1
                    elif 'yes' in img_lower:
                        label = 2
                    self.images.append((os.path.join(root_dir, img_name), label))
            print(f"Loaded {len(self.images)} validation images")
        else:
            print("Loading training data...")
            class_counts = {0: 0, 1: 0, 2: 0}
            for label, class_name in enumerate(self.classes):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append((os.path.join(class_dir, img_name), label))
                            class_counts[label] += 1

            print(f"Class distribution: {class_counts}")

            if len(set(class_counts.values())) > 1:
                print("Balancing classes...")
                image_paths = [x[0] for x in self.images]
                labels = [x[1] for x in self.images]
                ros = RandomOverSampler()
                _, labels = ros.fit_resample(np.array(image_paths).reshape(-1, 1), labels)
                self.images = list(zip(image_paths, labels))
                print(f"Balanced classes: {Counter(labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path
