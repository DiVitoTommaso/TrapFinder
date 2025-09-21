from environment import *


# ======================
# Vision Transformer Model for Coordinate Regression
# ======================
class TransformerNet(nn.Module):
    def __init__(self, img_size, num_points, hidden_dim, num_layers, drop_prob, drop_path_prob=0.2):
        super().__init__()
        self.num_points = num_points

        # Create a Vision Transformer (ViT) backbone from timm
        # Using tiny variant, pretrained=False (you can set True if weights available)
        self.vit = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=True,
            img_size=img_size,
            drop_rate=drop_prob,
            drop_path_rate=drop_path_prob
        )
        self.vit.head = nn.Identity()  # remove classification head → use features

        embed_dim = self.vit.num_features  # dimension of ViT output embeddings

        # Build a feed-forward decoder (MLP)
        layers = []
        in_dim = embed_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop_prob))
            in_dim = hidden_dim

        # Final layer outputs (num_points * 2) = (x,y) coordinates
        layers.append(nn.Linear(in_dim, num_points * 2))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        features = self.vit(x)  # extract image features with ViT
        out = self.decoder(features)  # decode to coordinates
        return out.view(B, self.num_points, 2)  # reshape to [batch, points, 2]


class ResNet(nn.Module):
    def __init__(self, img_size, num_points, hidden_dim, num_layers, drop_prob):
        super().__init__()
        self.num_points = num_points

        # Create a ResNet backbone
        self.resnet = timm.create_model(
            "resnet18",
            pretrained=True,
            num_classes=0,  # remove classification head → output features
            global_pool="avg"  # get pooled feature vector
        )

        embed_dim = self.resnet.num_features  # dimension of resnet output features

        # Build a feed-forward decoder (MLP)
        layers = []
        in_dim = embed_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop_prob))
            in_dim = hidden_dim

        # Final layer outputs (num_points * 2) = (x,y) coordinates
        layers.append(nn.Linear(in_dim, num_points * 2))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        features = self.resnet(x)  # extract image features with ResNet
        out = self.decoder(features)  # decode to coordinates
        return out.view(B, self.num_points, 2)  # reshape to [batch, points, 2]
