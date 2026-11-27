import { models as initialModels } from '../catalog.js?v=2';

const bilingual = (zh, en) => ({ zh, en });
const source = (zh, en, url) => ({ label: bilingual(zh, en), url });
const metric = (zh, en, value) => ({ label: bilingual(zh, en), value });
const galleryRoot = 'https://sebastianraschka.com/llm-architecture-gallery';
const original = 'https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490';

function classicModel(id, name, params, layers, width, heads, context) {
  const isThree = id === 'gpt3';
  const isXL = id === 'gpt2-xl';
  const hf = 'https://huggingface.co/openai-community/' + (isXL ? 'gpt2-xl' : 'gpt2');
  const scale = isThree
    ? bilingual(
        '按原版布局展示 96 层、每层 96 个头。图中使用 1,024 个位置和统一因果矩阵。GPT-3 论文的上下文为 2,048，并交替使用全注意力与局部带状稀疏注意力。OpenAI 未公开 175B 检查点，单元提供坐标、形状与计算公式。',
        'All 96 layers and 96 heads per layer follow the reference layout, with 1,024 displayed positions and a uniform causal matrix. The paper specifies a 2,048-token context and alternating dense and locally banded sparse attention. The 175B weights are unpublished. Cells show coordinates, shapes and equations.',
      )
    : bilingual(
        `完整展示 ${layers} 层和每层 ${heads} 个头。停在权重上时，浏览器从已固定版本的公开检查点只读取所需的一小段字节。激活值取决于具体输入，本视图提供它们的坐标、依赖和公式。`,
        `Inspect all ${layers} layers and ${heads} heads per layer. Hover over a weight to read a small byte range from the pinned public checkpoint. Activation cells show the equation and inputs needed to compute them. The browser does not run full-size GPT-2 inference.`,
      );
  return {
    id,
    name,
    short: name,
    layers,
    heads,
    width,
    displayParams: params,
    architecture: 'Causal Transformer',
    stats: [
      metric('参数量', 'Parameters', params),
      metric('解码层数', 'Decoder layers', String(layers)),
      metric('隐藏宽度', 'Hidden width', width.toLocaleString('en-US')),
      metric('上下文长度', 'Context length', context.toLocaleString('en-US')),
    ],
    scope: scale,
    gallery: isThree
      ? {
          url: 'https://arxiv.org/abs/2005.14165',
          label: bilingual('GPT-3 原论文 · 表 2.1', 'GPT-3 paper · Table 2.1'),
          note: bilingual(
            '这里链接 GPT-3 的原始配置。也可从下方进入 Gallery 浏览同族的 GPT-2 XL。',
            'Read the original GPT-3 specifications. The related GPT-2 XL diagram is linked below.',
          ),
        }
      : {
          url: galleryRoot + '/models/gpt-2-xl-1-5b/',
          label: isXL
            ? bilingual('GPT-2 XL · 同一模型', 'GPT-2 XL · exact model')
            : bilingual('GPT-2 XL · 同族二维图', 'GPT-2 XL · related architecture'),
          note: isXL
            ? bilingual(
                '对照位置嵌入、48 个解码块与词表投影。',
                'Compare the positional embeddings, 48 decoder blocks and vocabulary projection.',
              )
            : bilingual(
                'Gallery 收录 XL。Small 使用相同的解码结构，层数、头数与隐藏宽度不同。',
                'The gallery covers XL. Small uses the same decoder design with a different depth, head count and width.',
              ),
        },
    sources: isThree
      ? [
          source('GPT-3 论文', 'GPT-3 paper', 'https://arxiv.org/abs/2005.14165'),
          source('GPT-2 XL 同族二维图', 'Related GPT-2 XL diagram', galleryRoot + '/models/gpt-2-xl-1-5b/'),
          source('原版模型布局', 'Original model layout', original + '/src/llm/GptModelLayout.ts'),
        ]
      : [
          source('模型配置', 'Model configuration', hf + '/blob/main/config.json'),
          source('公开权重', 'Public checkpoint', hf + '/tree/main'),
          source('原版模型布局', 'Original model layout', original + '/src/llm/GptModelLayout.ts'),
        ],
  };
}
const nano = { ...initialModels[0], layers: 3, heads: 3, width: 48, displayParams: '85,728' };
const llama = { ...initialModels[1], layers: 32, heads: 32, width: 4096, displayParams: '8B' };
const qwen = { ...initialModels[2], layers: 64, heads: 24, width: 5120, displayParams: '27B' };
export const models = [
  classicModel('gpt2-small', 'GPT-2 Small', '124M', 12, 768, 12, 1024),
  nano,
  classicModel('gpt2-xl', 'GPT-2 XL', '1.56B', 48, 1600, 25, 1024),
  classicModel('gpt3', 'GPT-3', '175B', 96, 12288, 96, 2048),
  llama,
  qwen,
];

export function isModern(id) {
  return id === 'llama' || id === 'qwen';
}
export function isGPT2(id) {
  return id === 'gpt2-small' || id === 'gpt2-xl';
}
export const tabLabels = {
  'gpt2-small': 'GPT-2 (small)',
  nanogpt: 'nano-gpt',
  'gpt2-xl': 'GPT-2 (XL)',
  gpt3: 'GPT-3',
  llama: 'Llama 3.1 · 8B',
  qwen: 'Qwen3.8 · 27B',
};
const chapter = (stage, op, zh, en) => ({ stage, op, label: bilingual(zh, en) });
export function chapters(id) {
  if (isModern(id))
    return [
      chapter('embed', 'embed', '嵌入', 'Embedding'),
      id === 'qwen'
        ? chapter('delta', 'deltaState', '递归状态', 'Delta state')
        : chapter('norm', 'ln1.lnResid', 'RMSNorm', 'RMSNorm'),
      chapter(
        'attention',
        'attnMtxSm',
        id === 'qwen' ? '门控 GQA' : '分组查询',
        id === 'qwen' ? 'Gated GQA' : 'Grouped queries',
      ),
      chapter('ffn', 'mlpAct', 'SwiGLU', 'SwiGLU'),
      chapter('output', 'logitsSoftmax', '词表输出', 'Readout'),
    ];
  return [
    chapter('embed', 'residual0', '嵌入', 'Embedding'),
    chapter('norm', 'ln2.lnResid', '归一化', 'LayerNorm'),
    chapter(
      'attention',
      id === 'gpt3' || id === 'gpt2-xl' ? 'qkvWeightBlock' : 'qWeightBlock',
      'Q/K/V 权重',
      'Q/K/V weights',
    ),
    chapter('attention', 'attnMtxSm', '因果注意力', 'Causal attention'),
    chapter('attention', 'projWeight', '输出投影', 'Projection'),
    chapter('ffn', 'mlpAct', 'MLP 激活', 'MLP activation'),
    chapter('ffn', 'mlpResidual', '残差', 'Residual'),
    chapter('output', 'logitsSoftmax', '词表输出', 'Readout'),
  ];
}
export const upstream = original;
