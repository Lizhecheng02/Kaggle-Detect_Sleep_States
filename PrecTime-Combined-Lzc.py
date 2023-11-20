import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1d_block(
    in_channels,
    out_channels,
    kernel_size=5,
    stride=1,
    padding=2,
    dilation=1,
    maxpool=False,
    dropout=False
):
    layers = [
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
    ]
    if maxpool:
        layers.append(nn.MaxPool1d(kernel_size=2))
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return nn.Sequential(*layers)


def transformer_encoder_model(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    activation="relu"
):
    """
    创建一个标准的 Transformer 模型。

    参数:
    - d_model: 模型的特征维度
    - nhead: 多头注意力机制中的头数
    - num_encoder_layers: 编码器中的层次数量
    - dim_feedforward: 前馈网络模型的维度
    - dropout: Dropout比例
    - activation: 激活函数类型

    返回:
    - 一个 Transformer 模型
    """
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation
    )
    transformer_encoder = nn.TransformerEncoder(
        encoder_layer,
        num_layers=num_encoder_layers
    )
    return transformer_encoder


class PrecTime(nn.Module):
    def __init__(
        self,
        input_channels=8,
        hidden_channels=128,
        output_channels=128,
        left_fe_kernel_size=5,
        right_fe_kernel_size=5,
        pr_kernel_size=5,
        left_fe_padding=2,   # 根据输入自动调节
        right_fe_padding=2,
        pr_padding=2,
        left_fe_stride=1,   # 三个stride默认设置为1
        right_fe_stride=1,
        pr_stride=1,
        left_fe_dilation=1,
        right_fe_dilation=1,
        pr_dilation=1,
        sequence_length=1024,
        num_classes=3,
        chunks=6,
        num_left_fe_layers=3,
        num_right_fe_layers=3,
        fe_fc_dimension=64,
        lstm1_dimension=100,
        lstm2_dimension=100,
        n_head=2,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        activation="relu",
        encoder_type="transformer"
    ):
        super(PrecTime, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.left_fe_kernel_size = left_fe_kernel_size
        self.right_fe_kernel_size = right_fe_kernel_size
        self.pr_kernel_size = pr_kernel_size
        self.left_fe_padding = left_fe_padding
        self.right_fe_padding = right_fe_padding
        self.pr_padding = pr_padding
        self.left_fe_stride = left_fe_stride
        self.right_fe_stride = right_fe_stride
        self.pr_stride = pr_stride
        self.left_fe_dilation = left_fe_dilation
        self.right_fe_dilation = right_fe_dilation
        self.pr_dilation = pr_dilation
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.chunks = chunks
        self.num_left_fe_layers = num_left_fe_layers
        self.num_right_fe_layers = num_right_fe_layers
        self.fe_fc_dimension = fe_fc_dimension
        self.lstm1_dimension = lstm1_dimension
        self.lstm2_dimension = lstm2_dimension
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.encoder_type = encoder_type

        if self.encoder_type not in ["lstm", "transformer"]:
            print("Please enter the right type of encoder!!!")

        if self.left_fe_dilation * (self.left_fe_kernel_size - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.left_fe_padding = (
                self.left_fe_dilation * (self.left_fe_kernel_size - 1)) // 2

        if self.right_fe_dilation * (self.right_fe_kernel_size - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.right_fe_padding = (
                self.right_fe_dilation * (self.right_fe_kernel_size - 1)) // 2

        if self.pr_dilation * (self.pr_kernel_size - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.pr_padding = (self.pr_dilation *
                               (self.pr_kernel_size - 1)) // 2

        # 左侧特征提取分支
        feature_extraction1_layer = []
        feature_extraction1_layer.extend(
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.left_fe_kernel_size,
                padding=self.left_fe_padding,
                stride=self.left_fe_stride,
                dilation=self.left_fe_dilation,
                maxpool=True,
                dropout=True
            )
        )
        for i in range(self.num_left_fe_layers):
            feature_extraction1_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.left_fe_kernel_size,
                    padding=self.left_fe_padding,
                    stride=self.left_fe_stride,
                    dilation=self.left_fe_dilation
                )
            ])
        self.feature_extraction1 = nn.Sequential(
            *feature_extraction1_layer
        )

        # 右侧特征提取分支
        feature_extraction2_layer = []
        feature_extraction2_layer.extend(
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.right_fe_kernel_size,
                padding=self.right_fe_padding,
                stride=self.right_fe_stride,
                dilation=self.right_fe_dilation,
                maxpool=True,
                dropout=True
            )
        )
        for i in range(self.num_right_fe_layers):
            feature_extraction2_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.right_fe_kernel_size,
                    padding=self.right_fe_padding,
                    stride=self.right_fe_stride,
                    dilation=self.right_fe_dilation
                )
            ])
        self.feature_extraction2 = nn.Sequential(
            *feature_extraction2_layer
        )

        self.fc_after_fe = nn.Linear(
            self.hidden_channels * 2 *
            (self.sequence_length // self.chunks // 2), self.fe_fc_dimension
        )

        if self.encoder_type == "lstm":
            # 中间LSTM层
            self.context_detection1 = nn.LSTM(
                input_size=self.fe_fc_dimension,
                hidden_size=self.lstm1_dimension,
                num_layers=1,
                bidirectional=True
            )
            self.context_detection2 = nn.LSTM(
                input_size=self.lstm1_dimension * 2,
                hidden_size=self.lstm2_dimension,
                num_layers=1,
                bidirectional=True
            )

            self.encoder_output_dimension = self.lstm2_dimension * 2

        if self.encoder_type == "transformer":
            # 中间Transformer层
            self.transformer_encoder = transformer_encoder_model(
                d_model=self.fe_fc_dimension,
                nhead=self.n_head,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation
            )

            self.encoder_output_dimension = self.fe_fc_dimension

        self.inter_upsample = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(
            in_features=self.fe_fc_dimension,
            out_features=self.num_classes
        )

        self.inter_upsample_di = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks // 2,
            mode='nearest'
        )

        self.prediction_refinement = nn.Sequential(
            conv1d_block(
                in_channels=self.hidden_channels * 2 + self.encoder_output_dimension,
                out_channels=self.output_channels,
                kernel_size=self.pr_kernel_size,
                padding=self.pr_padding,
                stride=self.pr_stride,
                dilation=self.pr_dilation,
                maxpool=False,
                dropout=False
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1d_block(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=self.pr_kernel_size,
                padding=self.pr_padding,
                stride=self.pr_stride,
                dilation=self.pr_dilation,
                maxpool=False,
                dropout=True
            ),
            nn.Dropout(p=0.5)
        )

        self.fc_final = nn.Linear(self.output_channels, self.num_classes)

    def forward(self, x):
        origin_x = x

        if x.shape[-1] % self.chunks != 0:
            print(ValueError("Seq Length Should be Divided by Num_Chunks"))

        if x.shape[1] != self.input_channels:
            print(ValueError(
                "The Channel of Your Input should equal to Defined Input Channel"))

        if x.shape[-1] != self.sequence_length:
            print(ValueError(
                "The Length of Your Input should equal to Defined Seq Length"))

        print("The shape of input:", x.shape)

        x = x.transpose(0, 1).reshape(
            x.shape[1], -1, x.shape[2] // self.chunks
        ).transpose(0, 1)
        print("The shape put into feature extraction:", x.shape)

        features1 = self.feature_extraction1(x)
        print("The output shape from left feature extraction:", features1.shape)
        features2 = self.feature_extraction2(x)
        print("The output shape from right feature extraction:", features2.shape)
        features_combined = torch.cat((features1, features2), dim=1)
        print("The shape after the concate of two output:",
              features_combined.shape)

        features_combined_flat = features_combined.view(
            origin_x.shape[0], self.chunks, -1)
        print("The shape after the flatten of concat output:",
              features_combined_flat.shape)
        features_combined_flat = self.fc_after_fe(features_combined_flat)
        print("The shape after using fc to reduce dimension:",
              features_combined_flat.shape)

        if self.encoder_type == "lstm":
            context1, _ = self.context_detection1(features_combined_flat)
            print("The output shape after first LSTM:", context1.shape)
            context2, _ = self.context_detection2(context1)
            print("The output shape after second LSTM:", context2.shape)

        if self.encoder_type == "transformer":
            context2 = self.transformer_encoder(features_combined_flat)

        output1 = context2.permute(0, 2, 1)
        # print(output1.shape)
        output1 = self.inter_upsample(output1)
        print("The first output after upsample:", output1.shape)
        output1 = output1.permute(0, 2, 1)
        # print(output1.shape)
        output1 = self.inter_fc(output1)
        print("The first output after fc:", output1.shape)

        di = context2.permute(0, 2, 1)
        # print(di.shape)
        di = self.inter_upsample_di(di)
        print("The shape after upsampling Di:", di.shape)
        ui = features_combined.transpose(0, 1).reshape(
            features_combined.shape[1], origin_x.shape[0], -1
        ).transpose(0, 1)
        print("The shape after Reshaping Ui:", ui.shape)
        # ui = self.inter_upsample2(ui)
        # print(ui.shape)
        combine_ui_di = torch.cat([ui, di], dim=1)
        print("The shape after combining Ui and Di:", combine_ui_di.shape)

        final_output = self.prediction_refinement(combine_ui_di)
        print("The shape after prediction refinement:", final_output.shape)
        final_output = self.fc_final(final_output.permute(0, 2, 1))
        print("The final shape after fc:", final_output.shape)


Model = PrecTime(
    input_channels=4,
    hidden_channels=64,
    right_fe_kernel_size=5,
    pr_kernel_size=7,
    right_fe_stride=1,
    pr_stride=1,
    right_fe_dilation=4,
    pr_dilation=6,
    fe_fc_dimension=32,
    lstm1_dimension=64,
    lstm2_dimension=128,
    num_classes=3,
    sequence_length=720,
    chunks=4,
    encoder_type="transformer"
)
print(Model)

total_params = sum(p.numel() for p in Model.parameters())
print(f"Total parameters: {total_params}")

x = torch.randn(3, 4, 720)
output = Model(x)


# input_channels,
# hidden_channels=128,
# output_channels=128,
# left_fe_kernel_size=5,
# right_fe_kernel_size=5,
# pr_kernel_size=5,
# left_fe_padding=2,   # 根据输入自动调节
# right_fe_padding=2,
# pr_padding=2,
# left_fe_stride=1,   # 两个stride默认设置为1
# right_fe_stride=1,
# pr_stride=1,
# left_fe_dilation=1,
# right_fe_dilation=1,
# pr_dilation=1,
# sequence_length=1024,
# num_classes=3,
# chunks=6,
# num_left_fe_layers=3,
# num_right_fe_layers=3,
# fe_fc_dimension=64,
# lstm1_dimension=100,
# lstm2_dimension=100
