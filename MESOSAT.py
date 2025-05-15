class Trident(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
    
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(in_features, num_classes)

        vit_config = {
            "num_labels": num_classes,
            "hidden_dropout_prob": 0.2,
            "attention_probs_dropout_prob": 0.2,
            "ignore_mismatched_sizes": True
        }
        self.vit1 = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            **vit_config
        )
        self.vit1.vit.gradient_checkpointing = True

        vit_config.update({
            "hidden_dropout_prob": 0.3,
            "attention_probs_dropout_prob": 0.3
        })
        self.vit2 = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            **vit_config
        )
        self.vit2.vit.gradient_checkpointing = True

    def forward(self, x):
        cnn_out = self.cnn(x)
        
        with torch.cuda.amp.autocast():
            vit1_out = self.vit1(x).logits
            vit2_out = self.vit2(x).logits

        return cnn_out, vit1_out, vit2_out

    def MajorityV(self, x, inject_faults=False):
        with torch.no_grad(), torch.cuda.amp.autocast():
            cnn_out = self.cnn(x)
            vit1_out = self.vit1(x).logits
            vit2_out = self.vit2(x).logits

            cnn_preds = torch.argmax(cnn_out, 1).cpu().numpy()
            vit1_preds = torch.argmax(vit1_out, 1).cpu().numpy()
            vit2_preds = torch.argmax(vit2_out, 1).cpu().numpy()
            
            if inject_faults:
                if input("\nDestroy my hard work? (y/n): ").lower() == 'y':
                    cnn_preds = np.array([inject_fault(pred, "ResNet18") for pred in cnn_preds])
                    vit1_preds = np.array([inject_fault(pred, "ViT Model 1") for pred in vit1_preds])
                    vit2_preds = np.array([inject_fault(pred, "ViT Model 2") for pred in vit2_preds])

            print("\n MODEL PREDICTIONS (first image in batch):")
            print(f"ResNet18: {['no', 'partial', 'yes'][cnn_preds[0]]}")
            print(f"ViT-1: {['no', 'partial', 'yes'][vit1_preds[0]]}")
            print(f"ViT-2: {['no', 'partial', 'yes'][vit2_preds[0]]}")

            final_pred = []
            for cnn_p, vit1_p, vit2_p in zip(cnn_preds, vit1_preds, vit2_preds):
                majority_vote = Counter([cnn_p, vit1_p, vit2_p]).most_common(1)[0][0]
                final_pred.append(majority_vote)
            
            return np.array(final_pred), cnn_out, (vit1_out, vit2_out)

def show_majority_voter_diagram():
    """ensemble diagram"""
    diagram = r"""
    ┌───────────────────────┐       ┌───────────────────┐
    │                       │       │                   │
    │      Model 1          ├───────►                   │
    │                       │       │                   │
    └──────────┬────────────┘       │    MAJORITY       │
               │                    │     VOTER         │
    ┌──────────▼────────────┐       │                   │
    │                       ├───────►                   │
    │        Model 2        │       │                   │
    │                       │       │                   │
    └──────────┬────────────┘       └─────────▲─────────┘
               │                              │
    ┌──────────▼────────────┐                 │
    │                       ├─────────────────┘
    │       Model 3         │
    │                       │
    └───────────────────────┘
    """
    print("\nENSEMBLE ARCHITECTURE:")
    print(diagram)

def inject_fault(prediction, model_name):
    """fault injection"""
    fault_types = [
        lambda x: 0,  
        lambda x: 1,  
        lambda x: 2,  
        lambda x: random.choice([0,1,2]),  
        lambda x: (x + 1) % 3  
    ]
    
    fault = random.choice(fault_types)
    corrupted_pred = fault(prediction)
    
    print(f"\n=_= INJECTED FAULT into {model_name}:")
    print(f"Original prediction: {['no', 'partial', 'yes'][prediction]} -> Corrupted: {['no', 'partial', 'yes'][corrupted_pred]}")
    return corrupted_pred

def show_classification_report(val_preds, val_labels):
    """Voter results"""
    print("\n Worthiness stats:")
    for i, (pred, true) in enumerate(zip(val_preds, val_labels)):
        pred_label = ['no', 'partial', 'yes'][pred]
        true_label = ['no', 'partial', 'yes'][true]
        result = "🔥🔥" if pred == true else "sorry love😔"
        print(f"Image {i+1}: Predicted {pred_label} | Actual {true_label} {result}")

def print_training_progress(epoch, total_epochs, loss, acc, history):
    clear_output(wait=True)
    

    ziggurat = r"""
     _______
    |   𒀭  |
   _|_______|_
  | Ziggurat  |
 _|___________|_
|      Ur       |
|_______________|
 ~ ~ ~ ~ ~ ~ ~ ~
    """
    
    progress = int((epoch + 1) / total_epochs * 50)
    pos = progress % 20  
    
    print("Training YESSSSS💖💖:\n")
    for e, (l, a) in enumerate(history):
        print(f"Epoch {e+1}: Loss={l:.4f} | Accuracy={a:.2f}%{' <- rn' if e == epoch else ''}")
    
    print("\n" + " " * pos + ziggurat)
    print(f"\nProgress: [{'=' * progress}{' ' * (50 - progress)}]")
    print(f"live love laugh: Loss={loss:.4f} | Accuracy={acc:.2f}%")

def TrainTrident(model, train_loader, val_loader, epochs=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model = model.to(device)

    accum_steps = 4
    batch_size = 8
    history = []

    all_labels = [item[1] for item in train_loader.dataset.images]
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = torch.max(class_counts).float() / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = sum(criterion(o, labels) for o in outputs) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                train_loss += loss.item() * accum_steps

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                preds, _, _ = model.MajorityV(images.to(device), inject_faults=False)
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        min_length = min(len(val_preds), len(val_labels))
        acc = np.mean(np.array(val_labels[:min_length]) == np.array(val_preds[:min_length])) * 100
        history.append((train_loss/len(train_loader), acc))
        scheduler.step(acc)
        
        print_training_progress(epoch, epochs, train_loss/len(train_loader), acc, history)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(CheckpointBruv, "best_model.pt"))
            print(f"\nACCURACY IMPROVED, AUTOSAVE! {acc:.2f}%")

def generate_heatmaps(model, val_loader, Visualise_innit):

    val_images = []
    for images, labels, img_paths in val_loader:
        for i in range(len(images)):
            val_images.append({
                'image': images[i],
                'label': labels[i],
                'path': img_paths[i]
            })
    
    def show_heatmap(image_idx):
        try:
            device = next(model.parameters()).device
            model.eval()
            
            data = val_images[image_idx]
            image, label, img_path = data['image'], data['label'], data['path']
            
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )
            image = inv_normalize(image)
            rgb_img = image.numpy().transpose(1, 2, 0)
            rgb_img = np.clip(rgb_img, 0, 1)
            
            target_layer = model.cnn.layer4[-1].conv2
            cam = GradCAM(model=model.cnn, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device))
            visualisation = show_cam_on_image(rgb_img, grayscale_cam[0], use_rgb=True)
            
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_img)
            plt.title(f"Archaeological site <3\nLabel: {val_loader.dataset.classes[label]}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(visualisation)
            plt.title("Heatmap")
            plt.axis('off')
            
            plt.tight_layout()
            
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(Visualise_innit, f"{img_name}_heatmap.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
            
            print(f"Heatmap saved cok sukur: {save_path}")
        
        except Exception as e:
            print(f"kys there's an error: {str(e)}")
    
    image_options = [(f"{os.path.basename(val_images[i]['path'])} ({val_loader.dataset.classes[val_images[i]['label']]})", i) 
                    for i in range(len(val_images))]
    
    interact(show_heatmap, image_idx=widgets.Dropdown(
        options=image_options,
        value=0,
        description='Choose your character (site pls):',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    ))

if __name__ == "__main__":

    img_size = 224
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Reloading magazine blud...")
    train_dataset = Mesopotamia(root_dir="/content/drive/MyDrive/DATA", transform=train_transform)
    val_dataset = Mesopotamia(
        root_dir="/content/drive/MyDrive/DATA/validation",
        transform=val_transform,
        is_validation=True
    )

    batch_size = 8 if torch.cuda.is_available() else 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    print("\nBy the rivers of babylon, where we initialised the model...")
    model = Trident(num_classes=3)

    show_majority_voter_diagram()

    print("\nStarting training...")
    start_time = time.time()
    TrainTrident(model, train_loader, val_loader, epochs=6)
    print(f"\nyayyy training completed in {(time.time()-start_time)/60:.1f} minutes")

    model.eval()
    val_preds, val_labels = [], []
    
    fault_choice = input("\nDo you wanna mess up the system on purpose? (y/n): ").lower()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            preds, _, _ = model.MajorityV(
                images.to(next(model.parameters()).device),
                inject_faults=fault_choice == 'y'
            )
            val_preds.extend(preds)
            val_labels.extend(labels.numpy())
    
    if input("\nWould you like to see the result of your war crimes? (y/n): ").lower() == 'y':
        show_classification_report(val_preds, val_labels)

    print("\nValidation Results:")
    print(f"Accuracy: {np.mean(np.array(val_labels) == np.array(val_preds)):.2f}")
    print("\nWorthiness stats:")
    print(classification_report(val_labels, val_preds, target_names=['no', 'partial', 'yes']))

    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['no', 'partial', 'yes'],
                yticklabels=['no', 'partial', 'yes'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(Visualise_innit, 'confusion_matrix.png'))
    plt.close()

    worthiness_counts = Counter(val_preds)
    total = len(val_preds)
    print("\nExcavation Worthiness Distribution:")
    print(f"Worthy: {worthiness_counts[2]} ({worthiness_counts[2]/total:.1%})")
    print(f"Partial: {worthiness_counts[1]} ({worthiness_counts[1]/total:.1%})")
    print(f"Not Worthy: {worthiness_counts[0]} ({worthiness_counts[0]/total:.1%})")
    
    print(f"\nGenerating HEATTTmaps for {len(val_dataset)} validation images...")
    generate_heatmaps(model, val_loader, Visualise_innit)
