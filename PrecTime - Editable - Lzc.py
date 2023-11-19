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


class PrecTime(nn.Module):
    def __init__(
        self,
        input_channels=8,
        hidden_channels=128,
        output_channels=128,
        kernel_size1=5,
        kernel_size2=5,
        kernel_size3=5,
        padding1=2,   # 根据输入自动调节
        padding2=2,
        padding3=2,
        stride1=1,   # 三个stride默认设置为1
        stride2=1,
        stride3=1,
        dilation1=1,
        dilation2=1,
        dilation3=1,
        sequence_length=1024,
        num_classes=3,
        chunks=6,
        fe1_layers=3,
        fe2_layers=3,
        fc1_dimension=64,
        lstm1_dimension=100,
        lstm2_dimension=100,
        inter_fc_dimension=3
    ):
        super(PrecTime, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.kernel_size3 = kernel_size3
        self.padding1 = padding1
        self.padding2 = padding2
        self.padding3 = padding3
        self.stride1 = stride1
        self.stride2 = stride2
        self.stride3 = stride3
        self.dilation1 = dilation1
        self.dilation2 = dilation2
        self.dilation3 = dilation3
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.chunks = chunks
        self.fe1_layers = fe1_layers
        self.fe2_layers = fe2_layers
        self.fc1_dimension = fc1_dimension
        self.lstm1_dimension = lstm1_dimension
        self.lstm2_dimension = lstm2_dimension
        self.inter_fc_dimension = inter_fc_dimension

        if self.dilation1 * (self.kernel_size1 - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.padding1 = (self.dilation1 * (self.kernel_size1 - 1)) // 2

        if self.dilation2 * (self.kernel_size2 - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.padding2 = (self.dilation2 * (self.kernel_size2 - 1)) // 2

        if self.dilation3 * (self.kernel_size3 - 1) % 2 != 0:
            print(ValueError("Please re-input dilation, kernel_size!!!"))
        else:
            self.padding3 = (self.dilation3 * (self.kernel_size3 - 1)) // 2

        # 左侧特征提取分支
        feature_extraction1_layer = []
        feature_extraction1_layer.extend(
            conv1d_block(
                in_channels=self.input_channels,
                out_channels=self.hidden_channels,
                kernel_size=self.kernel_size1,
                padding=self.padding1,
                stride=self.stride1,
                dilation=self.dilation1,
                maxpool=True,
                dropout=True
            )
        )
        for i in range(self.fe1_layers):
            feature_extraction1_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size1,
                    padding=self.padding1,
                    stride=self.stride1,
                    dilation=self.dilation1
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
                kernel_size=self.kernel_size2,
                padding=self.padding2,
                stride=self.stride2,
                dilation=self.dilation2,
                maxpool=True,
                dropout=True
            )
        )
        for i in range(self.fe2_layers):
            feature_extraction2_layer.extend([
                conv1d_block(
                    in_channels=self.hidden_channels,
                    out_channels=self.hidden_channels,
                    kernel_size=self.kernel_size2,
                    padding=self.padding2,
                    stride=self.stride2,
                    dilation=self.dilation2
                )
            ])
        self.feature_extraction2 = nn.Sequential(
            *feature_extraction2_layer
        )

        self.fc1 = nn.Linear(
            self.hidden_channels * 2 *
            (self.sequence_length // self.chunks // 2), self.fc1_dimension
        )

        # 中间RNN层
        self.context_detection1 = nn.LSTM(
            input_size=self.fc1_dimension,
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

        self.inter_upsample = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks,
            mode='nearest'
        )
        self.inter_fc = nn.Linear(
            in_features=self.lstm2_dimension * 2,
            out_features=self.inter_fc_dimension
        )

        self.inter_upsample_di = nn.Upsample(
            scale_factor=self.sequence_length // self.chunks // 2,
            mode='nearest'
        )

        self.prediction_refinement = nn.Sequential(
            conv1d_block(
                in_channels=self.hidden_channels * 2 + self.lstm2_dimension * 2,
                out_channels=self.output_channels,
                kernel_size=self.kernel_size3,
                padding=self.padding3,
                stride=self.stride3,
                dilation=self.dilation3,
                maxpool=False,
                dropout=False
            ),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv1d_block(
                in_channels=self.output_channels,
                out_channels=self.output_channels,
                kernel_size=self.kernel_size3,
                padding=self.padding3,
                stride=self.stride3,
                dilation=self.dilation3,
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


Model = PrecTime(
    input_channels=4,
    hidden_channels=64,
    kernel_size2=5,
    kernel_size3=7,
    stride2=1,
    stride3=1,
    dilation2=4,
    dilation3=6,
    fc1_dimension=32,
    lstm1_dimension=64,
    lstm2_dimension=128,
    num_classes=3,
    sequence_length=720,
    chunks=4
)
print(Model)

total_params = sum(p.numel() for p in Model.parameters())
print(f"Total parameters: {total_params}")

x = torch.randn(3, 4, 720)
output = Model(x)


# input_channels,
# hidden_channels=128,
# output_channels=128,
# kernel_size1=5,
# kernel_size2=5,
# kernel_size3=5,
# padding1=2,   # 根据输入自动调节
# padding2=2,
# padding3=2,
# stride1=1,   # 两个stride默认设置为1
# stride2=1,
# stride3=1,
# dilation1=1,
# dilation2=1,
# dilation3=1,
# sequence_length=1024,
# num_classes=3,
# chunks=6,
# fe1_layers=3,
# fe2_layers=3,
# fc1_dimension=64,
# lstm1_dimension=100,
# lstm2_dimension=100,
# inter_fc_dimension=3
