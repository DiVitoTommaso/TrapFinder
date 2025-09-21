from environment import *


# -----------------------
# Resize and center-crop with pixel coordinates
# -----------------------
def resize_and_crop(image, centers, input_size):
    """Resize and center-crop the image, adjusting centers in pixel coordinates."""
    H_new, W_new = input_size

    # Resize image keeping aspect ratio (shorter side = H_new)
    resize_transform = transforms.Resize(H_new)
    image_resized = resize_transform(image)
    W_resized, H_resized = image_resized.size

    # Scale centers to resized image
    w, h = image.size
    centers_resized = []
    for x, y in centers:
        x_scaled = x * (W_resized / w)
        y_scaled = y * (H_resized / h)
        centers_resized.append([x_scaled, y_scaled])

    # Center crop
    crop_transform = transforms.CenterCrop((H_new, W_new))
    image_cropped = crop_transform(image_resized)

    # Crop offset
    x_offset = (W_resized - W_new) / 2
    y_offset = (H_resized - H_new) / 2

    # Adjust centers for cropping
    centers_cropped = []
    for x_scaled, y_scaled in centers_resized:
        x_crop = x_scaled - x_offset
        y_crop = y_scaled - y_offset
        centers_cropped.append([x_crop, y_crop])

    return image_cropped, torch.tensor(centers_cropped, dtype=torch.float32)


# -----------------------
# Count points in spatial buckets (pixel-based)
# -----------------------
def count_points_in_buckets(dataset, H_bins, W_bins, input_size):
    """Count points in each spatial bin, supports Dataset or Subset."""
    H, W = input_size
    counts = torch.zeros(H_bins, W_bins, dtype=torch.int32)

    def get_sample(idx):
        if isinstance(dataset, Subset):
            return dataset.dataset[dataset.indices[idx]]
        return dataset[idx]

    for idx in range(len(dataset)):
        _, centers = get_sample(idx)
        for x, y in centers:
            if x == 0 and y == 0:
                continue
            xi = min(int(x.item() / W * W_bins), W_bins - 1)
            yi = min(int(y.item() / H * H_bins), H_bins - 1)
            counts[yi, xi] += 1

    return counts


# -----------------------
# Dataset with augmentation
# -----------------------
def apply_augmentation(img, centers, input_size):
    """Random augmentations with pixel coordinates."""
    H, W = input_size
    img = TF.to_pil_image(img)

    # Horizontal flip
    if random.random() < 0.5:
        img = TF.hflip(img)
        centers[:, 0] = W - centers[:, 0]

    # Vertical flip
    if random.random() < 0.5:
        img = TF.vflip(img)
        centers[:, 1] = H - centers[:, 1]

    # Rotation around image center
    angle = random.uniform(-30, 30)
    img = TF.rotate(img, angle)
    angle_rad = -math.radians(angle)
    cx, cy = W / 2, H / 2
    x, y = centers[:, 0], centers[:, 1]
    x_rot = (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad) + cx
    y_rot = (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad) + cy
    centers[:, 0], centers[:, 1] = x_rot, y_rot

    return TF.to_tensor(img), torch.clamp(centers, 0.0, max(INPUT_SIZE))


class CenterRegressionDataset(Dataset):
    """Loads images and labeled centers, applies resizing, cropping, and optional augmentation."""

    def __init__(self, root_dir, label, points, input_size):
        """
        Args:
            root_dir (str): Root directory containing images and 'labels.json'.
            label (str): Target label to extract from annotations.
            points (int): Maximum number of points per image.
            input_size (int or tuple): Size to resize/crop images to.
        """
        self.samples = []            # List to store (image_tensor, centers) pairs
        self.label = label           # Label we are interested in
        self.points = points         # Max number of points per image
        self.input_size = input_size # Image rescaled (resize + crop) input size
        self.augmentation = False    # Flag to enable/disable online augmentation
        self.to_tensor = transforms.ToTensor()  # Convert PIL images to tensors

        # Iterate over all subdirectories and files in the root directory
        for subdir, _, files in os.walk(root_dir):
            print(f"Loading {subdir}")

            # Skip directories without labels.json
            if 'labels.json' not in files:
                continue

            # Load JSON labels
            with open(os.path.join(subdir, 'labels.json'), "r") as f:
                labels = json.load(f)

            label_keys = sorted(labels.keys())  # Sorted keys for consistency
            image_files = sorted(f for f in files if f.split(".")[-1].lower() in ['jpg', 'png'])

            # Filter images that do not appear in the label JSON
            images_with_traps = sorted(k.split("\\")[-1] for k in label_keys)
            images_no_traps = sorted([f for f in image_files if f.split("\\")[-1] not in images_with_traps])

            # Iterate over each labeled image
            for i in range(len(label_keys)):
                image_path = os.path.join(subdir, images_no_traps[i])
                image = Image.open(image_path).convert("RGB")  # Load and convert to RGB

                image_labels = labels[label_keys[i]]  # Get annotations for this image

                count = 0
                centers = []

                for sample in image_labels:
                    if sample['label'] == self.label:
                        centers.append(sample["center"])
                        count += 1
                        break  # Only keep first matching label

                # Pad with (0,0) if fewer points than self.points
                if count <= self.points:

                    # Bad Sample
                    if count == 0:
                        continue

                    for _ in range(self.points - count):
                        centers.append((0, 0))
                else:
                    # Skip images with too many points (bad samples)
                    continue
                centers = list(sorted(centers, key=lambda p: math.sqrt(p[0]**2 + p[1]**2), reverse=True))

                # Resize and crop image, adjust centers accordingly
                image_cropped, centers = resize_and_crop(image, centers, self.input_size)
                image_tensor = self.to_tensor(image_cropped)  # Convert to tensor

                # Append the sample to the dataset
                self.samples.append((image_tensor, centers))

        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a single sample (image_tensor, centers).
        Optionally applies online augmentation.
        """
        image_tensor, centers = self.samples[idx]

        # Apply augmentation with 10% probability
        if self.augmentation and torch.rand(1).item() < 0.1:
            image_tensor, centers = apply_augmentation(image_tensor.clone(), centers.clone(), self.input_size)

        return image_tensor, centers

    def enable_augmentation(self):
        """Enable online augmentation for future __getitem__ calls."""
        self.augmentation = True

    def disable_augmentation(self):
        self.augmentation = False


def smoter_oversample(dataset, num_samples, sigma=0.02):
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        indices = list(dataset.indices)  # copy to avoid modifying original in-place
    else:
        base_dataset = dataset
        indices = list(range(len(base_dataset)))

    # Extract images and targets
    imgs = [base_dataset.samples[i][0] for i in indices]
    centers = torch.stack([base_dataset.samples[i][1] for i in indices])

    new_samples = []
    for _ in range(num_samples):
        i, j = random.sample(range(len(indices)), 2)
        lam = random.random()
        new_center = lam * centers[i] + (1 - lam) * centers[j]
        noise = torch.randn_like(new_center) * sigma
        new_center = torch.clamp(new_center + noise, 0.0, max(base_dataset.input_size))
        new_samples.append((imgs[i], new_center))

    # Add new samples to the base dataset
    start_idx = len(base_dataset.samples)
    base_dataset.samples.extend(new_samples)

    if isinstance(dataset, torch.utils.data.Subset):
        # Update subset indices to include new ones
        new_indices = list(range(start_idx, start_idx + len(new_samples)))
        dataset.indices = list(dataset.indices) + new_indices
        return dataset
    else:
        # If it was a full dataset, just return it
        return base_dataset


# -----------------------
# Weighted sampler
# -----------------------
def make_weighted_sampler(dataset, h_bins, w_bins):
    """
    Create a weighted sampler for pixel coordinates.
    Supports Dataset or Subset.
    """
    H, W = INPUT_SIZE
    counts = torch.zeros(h_bins, w_bins, dtype=torch.int32)

    def get_sample(idx):
        if isinstance(dataset, Subset):
            return dataset.dataset[dataset.indices[idx]]
        return dataset[idx]

    # Count points in each bin
    for idx in range(len(dataset)):
        _, centers = get_sample(idx)
        for x, y in centers:
            if x == 0 and y == 0:
                continue
            xi = min(int(x.item() / W * w_bins), w_bins - 1)
            yi = min(int(y.item() / H * h_bins), h_bins - 1)
            counts[yi, xi] += 1

    counts = counts.float() + 1e-6
    weights = []

    for idx in range(len(dataset)):
        _, centers = get_sample(idx)
        x, y = centers[0]
        if x == 0 and y == 0:
            weights.append(0.0)
            continue
        xi = min(int(x.item() / W * w_bins), w_bins - 1)
        yi = min(int(y.item() / H * h_bins), h_bins - 1)
        weights.append(1.0 / counts[yi, xi].item())

    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=2 * len(weights), replacement=True)
    return sampler
