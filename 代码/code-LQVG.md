import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class ResNetMultiScale(nn.Module):
    # 多尺度视觉特征
    def __init__(self):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2  # stride 8
        self.layer3 = net.layer3  # stride 16
        self.layer4 = net.layer4  # stride 32

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c3, c4, c5]


class MSCMA(nn.Module):
    def __init__(self, hidden_dim=256, nheads=8):
        super().__init__()
        self.text_to_vision = nn.MultiheadAttention(hidden_dim, nheads)
        self.vision_to_text = nn.MultiheadAttention(hidden_dim, nheads)

    def forward(self, visual_tokens, text_tokens):
        # visual_tokens: list of [HW, B, C]
        # text_tokens: [L, B, C]

        refined_text = text_tokens
        refined_visual = []

        for v in visual_tokens:
            # LVI: language queries attend to visual tokens
            t2v, _ = self.text_to_vision(refined_text, v, v)
            refined_text = refined_text * t2v

            # VLI: visual tokens attend to original/refined language tokens
            v2t, _ = self.vision_to_text(v, text_tokens, text_tokens)
            refined_visual.append(v * v2t)

        return refined_visual, refined_text


class SimpleLQVG(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        nheads=8,
        num_queries=10,
        num_encoder_layers=4,
        num_decoder_layers=4,
        text_model_name="bert-base-uncased",
    ):
        super().__init__()

        self.backbone = ResNetMultiScale()

        self.input_proj = nn.ModuleList([
            nn.Conv2d(512, hidden_dim, 1),
            nn.Conv2d(1024, hidden_dim, 1),
            nn.Conv2d(2048, hidden_dim, 1),
        ])

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        # 词级文本特征
        self.text_proj = nn.Linear(768, hidden_dim)
        self.text_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.mscma = MSCMA(hidden_dim, nheads)

        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nheads)
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, nheads)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # 与 LQVG 思路一致：句子特征复制成多个 language queries，
        # 再加一个可学习 query embedding 区分不同候选框。
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))

        self.class_head = nn.Linear(hidden_dim, 1)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, images, input_ids, attention_mask):
        # 1. image encoder: multiscale visual features
        feats = self.backbone(images)

        visual_tokens = []
        for feat, proj in zip(feats, self.input_proj):
            x = proj(feat)                     # [B, C, H, W]
            x = x.flatten(2).permute(2, 0, 1)  # [HW, B, C]
            visual_tokens.append(x)

        # 2. text encoder: word-level text features
        text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state                    # [B, L, 768]

        text = self.text_proj(text)            # [B, L, C]
        text_tokens = text.permute(1, 0, 2)    # [L, B, C]

        # 3. MSCMA: multiscale cross-modal alignment
        visual_tokens, text_tokens = self.mscma(visual_tokens, text_tokens)

        # 4. multimodal DETR encoder over visual tokens
        memory = torch.cat(visual_tokens, dim=0)
        memory = self.encoder(memory)

        # 5. language queries from sentence feature
        cls_token = text_tokens[0]             # [B, C]
        sentence = self.text_pool(cls_token)   # [B, C]

        queries = sentence.unsqueeze(0) + self.query_embed.unsqueeze(1)
        hs = self.decoder(queries, memory)     # [Q, B, C]
        hs = hs.transpose(0, 1)                # [B, Q, C]

        # 6. heads: confidence + box
        logits = self.class_head(hs)           # [B, Q, 1]
        boxes = self.box_head(hs).sigmoid()    # [B, Q, 4], cxcywh normalized

        return logits, boxes


# Example
model = SimpleLQVG(hidden_dim=256, nheads=8, num_queries=10)
model.eval()

images = torch.randn(1, 3, 800, 800)
input_ids = torch.randint(0, 1000, (1, 16))
attention_mask = torch.ones(1, 16).long()

logits, boxes = model(images, input_ids, attention_mask)
best = logits.sigmoid().squeeze(-1).argmax(dim=1)
final_box = boxes[torch.arange(boxes.size(0)), best]

print(logits.shape)   # [1, 10, 1]
print(boxes.shape)    # [1, 10, 4]
print(final_box)      # highest-score predicted box