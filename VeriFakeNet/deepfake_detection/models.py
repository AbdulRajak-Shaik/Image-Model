import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class DeepfakeImageModel(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(DeepfakeImageModel, self).__init__()
        
        # We use EfficientNet-B3
        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            self.efficientnet = EfficientNet.from_name('efficientnet-b3')
            
        # Modify the last layer for binary classification (Real/Fake)
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

    def extract_features(self, x):
        """
        Extracts features before the final classification layer, useful for the BiLSTM.
        """
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x


class DeepfakeVideoModel(nn.Module):
    def __init__(self, feature_dim=1536, hidden_dim=512, num_layers=2, num_classes=1, pretrained_image_model=None):
        super(DeepfakeVideoModel, self).__init__()
        
        self.image_model = pretrained_image_model
        if self.image_model is None:
            self.image_model = DeepfakeImageModel(pretrained=True)
            
        # Freeze the image model if you only want to train the BiLSTM
        # for param in self.image_model.parameters():
        #     param.requires_grad = False
            
        # 1536 is the feature dimension out of EfficientNet-B3
        self.lstm = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Flatten batch and sequence dimensions for efficientnet
        x = x.view(batch_size * seq_len, c, h, w)
        
        with torch.no_grad(): # Typically freeze CNN when extracting features for LSTM to save memory
            features = self.image_model.extract_features(x)
            
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)
        
        # Pass through BiLSTM
        lstm_out, (hn, cn) = self.lstm(features)
        
        # Take the output of the last time step for both directions
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        final_features = lstm_out[:, -1, :] 
        
        # Final classification
        out = self.fc(final_features)
        return out
