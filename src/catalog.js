const text = (zh, en) => ({ zh, en });
const metric = (zh, en, value) => ({ label: text(zh, en), value });
const source = (zh, en, url) => ({ label: text(zh, en), url });

export const models = [
  {
    id: 'nanogpt',
    name: 'NanoGPT',
    short: 'NanoGPT',
    org: 'minGPT',
    architecture: 'Causal Transformer',
    subtitle: text(
      '从 A、B、C 中选六个字母，观察小型 Transformer 如何排序。',
      'Follow six letters through a tiny Transformer trained to put A, B and C in order.',
    ),
    stats: [
      metric('参数量', 'Parameters', '85,728'),
      { ...metric('解码层数', 'Decoder layers', '3'), shown: 1 },
      metric('隐藏宽度', 'Hidden width', '48'),
      metric('上下文长度', 'Context length', '11'),
    ],
    stages: [
      {
        id: 'embed',
        title: text('词元与位置嵌入', 'Token & position embeddings'),
        tag: 'Embedding',
        summary: text(
          '从词表中取出字母的 48 维向量，再加上当前位置的向量。两个 A 共用同一行字母嵌入，处在不同位置时，送进第一层的向量会不同。',
          'Look up a 48-value vector for the letter and add a learned vector for its position. Two A’s use the same letter vector, yet reach the first layer with different inputs.',
        ),
        formula: 'xₜ = E[tokenₜ] + P[t]',
        details: [
          metric('字母嵌入表', 'Token table', '3 × 48'),
          metric('位置嵌入表', 'Position table', '11 × 48'),
        ],
      },
      {
        id: 'norm',
        title: text('层归一化', 'Layer normalization'),
        tag: 'LayerNorm',
        summary: text(
          '在一个位置的 48 个通道上减去均值，再按标准差缩放。注意力与前馈网络各有一套归一化参数，两条支路的输出分别加回残差流。',
          'Center and rescale the 48 channels at each position. Attention and the feed-forward network each get their own normalization and add their updates back to the residual stream.',
        ),
        formula: 'LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β',
        details: [
          metric('归一化宽度', 'Channels per token', '48'),
          metric('每层归一化次数', 'Norms per block', '2'),
        ],
      },
      {
        id: 'attention',
        title: text('因果多头注意力', 'Causal self-attention'),
        tag: 'MHA',
        summary: text(
          '每个头把当前位置的查询与当前及更早位置的键比对。因果掩码屏蔽未来字母，三个头分别加权读取值，再将读出的特征合并。',
          'Each query scores keys at its own and earlier positions. A causal mask blocks future letters. Three heads read weighted values independently, then combine their features.',
        ),
        formula: 'A = softmax(QKᵀ / √16 + mask)',
        details: [
          metric('查询头 / 键值头', 'Query / KV heads', '3 / 3'),
          metric('每头宽度', 'Head width', '16'),
        ],
      },
      {
        id: 'ffn',
        title: text('GELU 前馈网络', 'GELU feed-forward'),
        tag: 'MLP',
        summary: text(
          '每个位置独立经过同一个网络。48 个通道扩成 192 个，经过 GELU，再压回 48 维，加到该位置的残差流上。',
          'A shared network transforms each position separately. It expands 48 channels to 192, applies GELU, then projects back to 48 and adds the result to the residual stream.',
        ),
        formula: 'FFN(u) = down(GELU(up(u)))',
        details: [
          metric('特征宽度', 'Channel widths', '48 → 192 → 48'),
          metric('激活函数', 'Activation', 'GELU'),
        ],
      },
      {
        id: 'output',
        title: text('下一个字母', 'Next-letter probabilities'),
        tag: 'Readout',
        summary: text(
          '经过三层和最终 LayerNorm 后，最后一个位置为 A、B、C 各给出一个分数。Softmax 得到三者的概率，每生成一个字母，就把它接回序列再预测下一个。',
          'After three blocks and a final LayerNorm, the last position scores A, B and C. Softmax turns the scores into probabilities. Each generated letter joins the sequence for the next prediction.',
        ),
        formula: 'p(next) = softmax(W_out · LN(xₜ))',
        details: [
          metric('输出投影', 'Output matrix', '3 × 48'),
          metric('输出字母表', 'Output alphabet', 'A / B / C'),
        ],
      },
    ],
    scope: text(
      '原演示用 minGPT 训练六字母排序。图中展开一个解码块，原模型依次执行三块。六个输入加上前五个输出，恰好用满 11 个上下文位置。',
      'Bycroft’s minGPT demo sorts six letters. The view opens one of three decoder blocks. Six inputs and the first five outputs fill the 11 positions needed to predict the last letter.',
    ),
    gallery: {
      url: 'https://sebastianraschka.com/llm-architecture-gallery/models/gpt-2-xl-1-5b/',
      label: text('GPT-2 XL · 同族架构', 'GPT-2 XL · family reference'),
      note: text(
        'Gallery 的 GPT-2 XL 图可对照位置嵌入、LayerNorm 和多头注意力。NanoGPT 的实际尺寸列在本站。',
        'The GPT-2 XL diagram shows the same building blocks at a larger scale. Use this atlas for the sorting model’s dimensions.',
      ),
    },
    sources: [
      source('Bycroft 的原版 3D 演示', 'Bycroft’s original 3D walkthrough', 'https://bbycroft.net/llm'),
      source(
        '演示权重与配置',
        'Demo weights and configuration',
        'https://github.com/bbycroft/llm-viz/blob/main/public/gpt-nano-sort-model.json',
      ),
      source(
        'minGPT 模型实现',
        'minGPT model implementation',
        'https://github.com/karpathy/minGPT/blob/master/mingpt/model.py',
      ),
    ],
  },
  {
    id: 'llama',
    name: 'Llama-3.1-8B-Instruct',
    short: 'Llama 3.1',
    org: 'Meta',
    architecture: 'Dense Transformer',
    subtitle: text(
      '沿一个解码块追踪指令模型，观察四个查询头怎样共享一组键和值。',
      'Trace one decoder block and see how four query heads share a single pair of key and value heads.',
    ),
    stats: [
      metric('参数量', 'Parameters', '8B'),
      { ...metric('解码层数', 'Decoder layers', '32'), shown: 1 },
      metric('隐藏宽度', 'Hidden width', '4,096'),
      metric('上下文长度', 'Context length', '131,072'),
    ],
    stages: [
      {
        id: 'embed',
        title: text('词元嵌入', 'Token embeddings'),
        tag: 'Embedding',
        summary: text(
          '分词器把文本编码成词元 ID，每个 ID 从嵌入表中选出一行 4,096 维向量。位置信息稍后通过注意力中的旋转位置编码进入计算。',
          'The tokenizer produces IDs. Each ID selects one row of 4,096 features from the embedding table. Position enters later when rotary embeddings act inside attention.',
        ),
        formula: 'xₜ = E[tokenₜ]',
        details: [
          metric('嵌入表', 'Embedding table', '128,256 × 4,096'),
          metric('位置编码', 'Position encoding', 'RoPE'),
        ],
      },
      {
        id: 'norm',
        title: text('均方根归一化', 'RMS normalization'),
        tag: 'RMSNorm',
        summary: text(
          '对每个词元计算通道的均方根，以它缩放向量，再乘以可学习的权重。每个解码块在注意力和前馈网络之前各做一次 RMSNorm。',
          'Scale each token vector by the root mean square of its channels, then apply learned gains. Each decoder block has one RMSNorm before attention and another before its feed-forward network.',
        ),
        formula: 'RMSNorm(x) = γ ⊙ x / √(mean(x²) + ε)',
        details: [
          metric('归一化宽度', 'Channels per token', '4,096'),
          metric('稳定项 ε', 'Epsilon', '10⁻⁵'),
        ],
      },
      {
        id: 'attention',
        title: text('分组查询注意力', 'Grouped-query attention'),
        tag: 'GQA',
        summary: text(
          'RoPE 旋转查询与键来编码相对位置。32 个查询头分成 8 组，每组 4 个头共享同一对键值头，因此生成时只需缓存 8 组键和值。',
          'RoPE rotates queries and keys to encode relative position. The 32 query heads form eight groups. Four queries share each pair of key and value heads, so decoding caches eight KV pairs per token.',
        ),
        formula: '32 Q ÷ 8 KV = 4 Q / KV',
        details: [
          metric('查询头 / 键值头', 'Query / KV heads', '32 / 8'),
          metric('每头宽度', 'Head width', '128'),
        ],
      },
      {
        id: 'ffn',
        title: text('SwiGLU 前馈网络', 'SwiGLU feed-forward'),
        tag: 'SwiGLU',
        summary: text(
          '输入分成两个投影，其中一支经过 SiLU，逐通道调节另一支。相乘的结果从 14,336 维投回 4,096 维，随后与残差相加。',
          'Two projections expand the input. SiLU on the gate branch controls the other branch channel by channel. The product projects from 14,336 features back to 4,096 and rejoins the residual.',
        ),
        formula: 'FFN(u) = down(SiLU(gate(u)) ⊙ up(u))',
        details: [
          metric('特征宽度', 'Channel widths', '4,096 → 14,336 → 4,096'),
          metric('门激活', 'Gate activation', 'SiLU'),
        ],
      },
      {
        id: 'output',
        title: text('词表投影', 'Vocabulary readout'),
        tag: 'LM head',
        summary: text(
          '第 32 个解码块输出的向量经过最终 RMSNorm。输出矩阵为 128,256 个词元分别打分，最后一个位置的分数用于选择回复中的下一个词元。',
          'After block 32, a final RMSNorm prepares the hidden state. The output matrix scores 128,256 vocabulary entries. The last position’s scores guide selection of the next reply token.',
        ),
        formula: 'logitsₜ = W_out · RMSNorm(xₜ)',
        details: [
          metric('输出投影', 'Output matrix', '128,256 × 4,096'),
          metric('最后一层', 'Last decoder block', '32 / 32'),
        ],
      },
    ],
    scope: text(
      '图中展开 32 个解码块中的一个，以少量头代表 32∶8 的查询与键值分组。标注给出原模型尺寸。Instruct 的指令微调沿用同一组解码构件。',
      'One representative block stands for 32. A few heads illustrate the 32-to-8 query/KV grouping. Labels give the original dimensions. Instruction tuning uses the same decoder structure.',
    ),
    gallery: {
      url: 'https://sebastianraschka.com/llm-architecture-gallery/models/llama-3-8b/',
      label: text('Llama 3 8B · 同族架构', 'Llama 3 8B · family reference'),
      note: text(
        'Gallery 收录的是 Llama 3 8B，可对照 GQA 与 SwiGLU。此处的 3.1 Instruct 将上下文从 8,192 扩展到 131,072 个词元。',
        'The gallery covers Llama 3 8B, which shares GQA and SwiGLU. Its 8,192-token context expands to 131,072 in this 3.1 Instruct model.',
      ),
    },
    sources: [
      source(
        'Meta 指令模型卡',
        'Meta Instruct model card',
        'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct',
      ),
      source(
        'Meta 技术报告 · 表 3',
        'Meta technical report · Table 3',
        'https://arxiv.org/html/2407.21783#S3.T3',
      ),
      source(
        'torchtune 的 8B 配置',
        '8B configuration in torchtune',
        'https://github.com/pytorch/torchtune/blob/main/torchtune/models/llama3_1/_model_builders.py',
      ),
    ],
  },
  {
    id: 'qwen',
    name: 'Qwen3.8-27B',
    short: 'Qwen3.8',
    org: 'Qwen Team',
    architecture: 'Dense Hybrid',
    subtitle: text(
      '沿四层混合解码器，看定长记忆如何与全注意力交替处理上下文。',
      'Follow a four-layer cycle that alternates fixed-size recurrent memory with full attention.',
    ),
    stats: [
      metric('参数量', 'Parameters', '27B'),
      { ...metric('语言解码层数', 'Text decoder layers', '64'), shown: 4 },
      metric('隐藏宽度', 'Hidden width', '5,120'),
      metric('原生上下文', 'Native context', '262,144'),
    ],
    stages: [
      {
        id: 'embed',
        title: text('文本嵌入', 'Text embeddings'),
        tag: 'Embedding',
        summary: text(
          '每个文本词元映射成 5,120 维的输入向量。完整模型还可接收图像和视频，本视图沿文本输入追踪语言解码器。',
          'Each text token maps to 5,120 input features. The full model also accepts images and video. This path follows a text input through the language decoder.',
        ),
        formula: 'xₜ = E[tokenₜ]',
        details: [
          metric('嵌入表（补齐后）', 'Padded embedding table', '248,320 × 5,120'),
          metric('原生上下文', 'Native context', '262,144'),
        ],
      },
      {
        id: 'delta',
        title: text('门控 DeltaNet', 'Gated DeltaNet'),
        tag: 'DeltaNet ×3',
        summary: text(
          '每组前三层各自维护定长状态。衰减门保留部分旧记忆，写入门用当前值与记忆读数的差来修正状态。每一层随后都运行自己的稠密前馈网络。',
          'Each of the first three layers keeps its own fixed-size state. A decay gate retains part of the memory. A write gate updates it from the difference between the current value and the remembered value. Each layer then runs its own dense FFN.',
        ),
        formula: 'Sₜ = DeltaUpdate(Sₜ₋₁, kₜ, vₜ, decay, βₜ)',
        details: [
          metric('QK 头 / V 头', 'QK / value heads', '16 / 48'),
          metric('每头宽度', 'Head width', '128'),
        ],
      },
      {
        id: 'attention',
        title: text('门控全注意力', 'Gated full attention'),
        tag: 'Gated GQA',
        summary: text(
          '每组第四层让查询直接读取所有已见位置。24 个查询头共享 4 组键值，每头 256 维中的 64 维使用 RoPE。输出门调节读出的特征，再交给本层的前馈网络。',
          'The fourth layer can read every earlier position directly. Its 24 query heads share four KV heads. RoPE rotates 64 of each head’s 256 channels. An output gate modulates the readout before the layer’s FFN.',
        ),
        formula: '24 Q ÷ 4 KV = 6 Q / KV',
        details: [
          metric('查询头 / 键值头', 'Query / KV heads', '24 / 4'),
          metric('旋转维度 / 每头宽度', 'Rotary / head dimensions', '64 / 256'),
        ],
      },
      {
        id: 'ffn',
        title: text('逐层稠密前馈', 'Dense FFN in every layer'),
        tag: 'SwiGLU',
        summary: text(
          '无论该层采用 DeltaNet 还是全注意力，每个词元都会经过本层的 SwiGLU。混合器和前馈网络各做一次 RMSNorm，并把各自的更新加回残差流。',
          'Every token passes through that layer’s SwiGLU, after either DeltaNet or full attention. Both the token mixer and the FFN normalize their input with RMSNorm and add their update to the residual stream.',
        ),
        formula: 'FFN(u) = down(SiLU(gate(u)) ⊙ up(u))',
        details: [
          metric('特征宽度', 'Channel widths', '5,120 → 17,408 → 5,120'),
          metric('稠密前馈层数', 'Dense FFN layers', '64'),
        ],
      },
      {
        id: 'output',
        title: text('语言模型输出', 'Language-model readout'),
        tag: 'LM head',
        summary: text(
          '四层一组共运行 16 次。语言主干经过最终 RMSNorm，把最后位置的向量映射到补齐后的词表。完整模型还带有一个多词元预测（MTP）层。',
          'The four-layer cycle runs 16 times. A final RMSNorm and the language head map the last position to the padded vocabulary. The full model also includes one multi-token prediction (MTP) layer.',
        ),
        formula: 'logitsₜ = W_out · RMSNorm(xₜ)',
        details: [
          metric('输出投影（补齐后）', 'Padded output matrix', '248,320 × 5,120'),
          metric('DeltaNet 层 / 全注意力层', 'DeltaNet / full-attention layers', '48 / 16'),
        ],
      },
    ],
    scope: text(
      '文字主干以四层为一组，重复 16 次。前三层各用 DeltaNet，第四层用门控全注意力，每层各有一个稠密 FFN。视图省略视觉编码器和额外的 MTP 层。',
      'The text backbone repeats a four-layer cycle 16 times. Three DeltaNet layers precede one gated full-attention layer, each with its own dense FFN. The view omits the vision encoder and auxiliary MTP layer.',
    ),
    gallery: {
      url: 'https://sebastianraschka.com/llm-architecture-gallery/models/qwen3-8-27b/',
      label: text('Qwen3.8 27B · 同一模型', 'Qwen3.8 27B · exact model'),
      note: text(
        'Gallery 的同款二维图展开了 DeltaNet 与门控全注意力两条路径，可与这里的四层循环对照。',
        'The matching diagram details both mixer types. Follow them alongside the four-layer cycle in this view.',
      ),
    },
    sources: [
      source('Qwen 官方模型卡', 'Qwen model card', 'https://huggingface.co/Qwen/Qwen3.8-27B'),
      source(
        'Qwen 官方配置',
        'Qwen model configuration',
        'https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json',
      ),
      source(
        'Transformers 参考实现',
        'Transformers reference implementation',
        'https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py',
      ),
    ],
  },
];
