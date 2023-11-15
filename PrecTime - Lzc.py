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
        # self.fe1out_shape = 0
        # self.fe2out_shape = 0

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
        self.padding = 8
        feature_extraction2_layer = []
        feature_extraction2_layer.extend([
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
                stride=self.stride,
                dilation=4
            ),
            conv1d_block(
                in_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.padding,
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
                    padding=self.padding,
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
            256 * (self.sequence_length // self.chunks // 2), 64
        )

        # 中间RNN层
        self.context_detection1 = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            bidirectional=True
        )
        self.context_detection2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            bidirectional=True
        )
        self.inter_upsample = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(in_features=256, out_features=3)

        self.prediction_refinement = nn.Sequential(
            conv1d_block(
                in_channels=input_channels,
                out_channels=128,
                maxpool=False,
                dropout=False
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1d_block(
                in_channels=input_channels,
                out_channels=128,
                maxpool=False,
                dropout=True
            )
        )

        # self.fc_final = nn.Linear(400, num_classes)

    def forward(self, x):
        if x.shape[-1] % self.chunks != 0:
            print(ValueError("Sequence_Length Should be Divided by Num_Chunks"))
        x = x.reshape(
            -1,
            self.input_channels,
            x.shape[-1] // self.chunks
        )
        print(x.shape)

        features1 = self.feature_extraction1(x)
        print(features1.shape)
        features2 = self.feature_extraction2(x)
        print(features2.shape)
        features_combined = torch.cat((features1, features2), dim=1)
        print(features_combined.shape)

        features_combined_flat = features_combined.view(1, self.chunks, -1)
        print(features_combined_flat.shape)
        features_combined_flat = self.fc1(features_combined_flat)
        print(features_combined_flat.shape)

        context1, _ = self.context_detection1(features_combined_flat)
        print(context1.shape)
        context2, _ = self.context_detection2(context1)
        print(context2.shape)

        output1 = context2.permute(0, 2, 1)
        print(output1.shape)
        output1 = self.inter_upsample(output1)
        print(output1.shape)
        output1 = output1.permute(0, 2, 1)
        print(output1.shape)
        output1 = self.inter_fc(output1)
        print(output1.shape)

        # context_upsampled = self.inter_upsample(context2)
        # print(context_upsampled.shape)

        # output1 = self.inter_fc(context_upsampled)
        # print(output1.shape)


Model = PrecTime(
    input_channels=32,
    num_classes=3,
    sequence_length=1024,
    chunks=4
)
print(Model)
x = torch.randn(1, 32, 1024)
output = Model(x)
