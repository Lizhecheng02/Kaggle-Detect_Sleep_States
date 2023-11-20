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
    layers = [nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )]
    if maxpool:
        layers.append(nn.MaxPool1d(kernel_size=2))
    if dropout:
        layers.append(nn.Dropout(p=0.5))
    return nn.Sequential(*layers)


class PrecTime(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels=128,
        kernel_size=5,
        padding=2,
        stride=1,
        dilation=1,
        sequence_length=1024,
        num_classes=3,
        chunks=6,
        fe1_layers=4,
        fe2_layers=4
    ):
        super(PrecTime, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.chunks = chunks
        self.fe1_layers = fe1_layers
        self.fe2_layers = fe2_layers

        # 左侧特征提取分支
        feature_extraction1_layer = []
        feature_extraction1_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe1_layers):
            feature_extraction1_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation
                )
            ])
        self.feature_extraction1 = nn.Sequential(
            *feature_extraction1_layer
        )

        # 计算通过fe1输出的序列长度
        # self.fe1out_shape = self.sequence_length // self.chunks
        # self.fe1out_shape = self.fe1out_shape - 5 + 1
        # self.fe1out_shape = self.fe1out_shape - 5 + 1
        # self.fe1out_shape = self.fe1out_shape // 2
        # for i in range(self.fe1_layers):
        #     self.fe1out_shape = self.fe1out_shape - 5 + 1
        # print("The Final Dimension of FE1 is:", self.fe1out_shape)

        # 右侧特征提取分支
        feature_extraction2_layer = []
        feature_extraction2_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=8,
                stride=self.stride,
                dilation=4
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=8,
                stride=self.stride,
                dilation=4,
                maxpool=True,
                dropout=True
            )
        ])
        for i in range(self.fe2_layers):
            feature_extraction2_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding=8,
                    stride=self.stride,
                    dilation=4
                )
            ])
        self.feature_extraction2 = nn.Sequential(
            *feature_extraction2_layer
        )

        # 计算通过fe2输出的序列长度
        # self.fe2out_shape = self.sequence_length // self.chunks
        # self.fe2out_shape = self.fe2out_shape - 17 + 1
        # self.fe2out_shape = self.fe2out_shape - 17 + 1
        # self.fe2out_shape = self.fe2out_shape // 2
        # for i in range(self.fe2_layers):
        #     self.fe2out_shape = self.fe2out_shape - 17 + 1
        # print("The Final Dimension of FE2 is:", self.fe2out_shape)

        # self.feout_shape = self.hidden_channels * \
        #     (self.fe1out_shape + self.fe2out_shape)

        self.fc1 = nn.Linear(
            self.hidden_channels * 2 *
            (self.sequence_length // self.chunks // 2), 64
        )

        # 中间LSTM层
        self.context_detection1 = nn.LSTM(
            input_size=64,
            hidden_size=100,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.context_detection2 = nn.LSTM(
            input_size=200,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.inter_upsample = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(
            in_features=self.context_detection2.hidden_size * 2,
            out_features=3
        )

        self.inter_upsample_di = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks // 2,
            mode='nearest'
        )
        # self.inter_upsample_ui = nn.Upsample(
        #     scale_factor=2,
        #     mode='nearest'
        # )

        self.prediction_refinement = nn.Sequential(
            conv1d_block(
                in_channels=self.hidden_channels * 2 + self.context_detection2.hidden_size * 2,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=2,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=False,
                dropout=False
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=2,
                stride=self.stride,
                dilation=self.dilation,
                maxpool=False,
                dropout=True
            ),
            nn.Dropout(p=0.5)
        )

        self.fc_final = nn.Linear(self.hidden_channels, num_classes)

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

        x = x.reshape(
            -1,
            self.input_channels,
            x.shape[-1] // self.chunks
        )
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
        features_combined_flat = self.fc1(features_combined_flat)
        print("The shape after using fc to reduce dimension:",
              features_combined_flat.shape)

        context1, _ = self.context_detection1(features_combined_flat)
        print("The output shape after first LSTM:", context1.shape)
        context2, _ = self.context_detection2(context1)
        print("The output shape after second LSTM:", context2.shape)

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

        return final_output


Model = PrecTime(
    input_channels=32,
    hidden_channels=64,
    num_classes=3,
    sequence_length=720,
    chunks=8
)
print(Model)

total_params = sum(p.numel() for p in Model.parameters())
print(f"Total parameters: {total_params}")

x = torch.randn(3, 32, 720)
output = Model(x)
