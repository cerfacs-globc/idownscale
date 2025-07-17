import sys
sys.path.append('.')

import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.networks.blocks import UnetrBasicBlock

class MiniSwinUNETR(SwinUNETR):
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        feature_size=24,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=2,
        downsample="merging",
        use_v2=False,
    ):
        # Modifier depths et num_heads pour réduire la profondeur
        depths = (2, 2, 2, 2)  # Moins de niveaux
        num_heads = (3, 6, 12, 24)  # Ajusté en conséquence
        
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            depths=depths,
            num_heads=num_heads,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )

        self.encoder5 = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=8 * feature_size,
                out_channels=8 * feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )

   
    def forward(self, x_in):
        if not torch.jit.is_scripting():
            self._check_input_size(x_in.shape[2:])
        
        hidden_states_out = self.swinViT(x_in, self.normalize)
        
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        dec3 = self.encoder5(hidden_states_out[3])
        dec2 = self.decoder4(dec3, hidden_states_out[2])
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        
        logits = self.out(out)
        return logits


if __name__=='__main__':
    model = MiniSwinUNETR(img_size=(160,160), in_channels=2, out_channels=1,spatial_dims=2)
    model = model.float()
    x = np.random.rand(1,2,160,160)
    x = torch.tensor(x)
    y_hat = model(x.float())
    print(y_hat.shape)