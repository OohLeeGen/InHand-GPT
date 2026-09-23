# InHand-GPT

用 **GPT-as-Policy 决策循环**驱动 MuJoCo 仿真中的 Shadow Dexterous Hand 完成**掌内方块旋转**（in-hand reorientation，`HandManipulateBlockRotateZ-v1`）任务的最小但真实可用的项目核心。

决策模型两档：

- **规则基线**（`RuleBaselinePolicy`）：已实现并跑通验证（本仓库内 5 局实测，见 DESIGN.md 第 5 节）；
- **GPT**（`GPTPolicy`）：OpenAI 兼容 API 客户端，配置即可用；**本次未做真实 API 调用**（key 后补），提供 `--dry-run` 打印完整 prompt。

## 架构

```
观测打包(observe) → 决策(policy: baseline|gpt) → 基元执行(primitives, 12–30 env步)
      ↑                                                    │
      └────────── 再观测 + 记录(record/loop) ←─────────────┘
```

- `inhand_gpt/observe.py` — 观测契约：四元数 → 带符号绕 Z 角误差（度）、漂移、成功/掉落判定
- `inhand_gpt/primitives.py` — Shadow Hand 基元库（全部经实测筛选，数据见文件头与 DESIGN.md）
- `inhand_gpt/policy.py` — 策略抽象 + 规则基线 + GPT 策略（含可离线单测的响应解析器）
- `inhand_gpt/loop.py` — 闭环：观测→决策→执行→再观测→记录/判定
- `inhand_gpt/record.py` — 逐周期记录 + JSON 导出
- `scripts/run_sim.py` — CLI 主入口；`scripts/tune_primitives.py` — 基元调参/验证工具

## 安装

```bash
conda activate inhand-sim           # 或任何 Python>=3.10 环境
pip install -r requirements-sim.txt # gymnasium-robotics, httpx, imageio-ffmpeg, imageio
pip install -e .                    # 可编辑安装（包名 inhand-gpt）
```

## 运行

规则基线（5 局 + 逐局 MP4/JSON）：

```bash
python scripts/run_sim.py --policy baseline --seed 0 --episodes 1 --max-cycles 25 \
    --video runs/seed0.mp4 --export runs/seed0.json
# 批量：--seed 0 --episodes 5（--export 自动按种子分文件）
```

GPT 策略（dry-run，打印完整 prompt 不发请求）：

```bash
python scripts/run_sim.py --policy gpt --dry-run --seed 0 --max-cycles 1
```

GPT 解析器离线自测：

```bash
python -m inhand_gpt.policy     # 7/7 checks passed
```

基元调参/验证：

```bash
python scripts/tune_primitives.py --scan all --seed 0          # 候选波形扫描
python scripts/tune_primitives.py --scan pkg --seeds 0 1 2 3 4 # 已固化基元库跨种子验证
```

## GPT API key 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `INHAND_API_KEY` | （无） | **必需**；缺失时 GPT 策略立即报清晰错误退出（exit 2），**绝不静默回退规则基线**（研究完整性：日志里每个决策都可追溯到所声明的策略） |
| `INHAND_API_BASE` | `https://api.openai.com/v1` | OpenAI 兼容端点 |
| `INHAND_MODEL` | `gpt-6` | 模型名 |

```bash
export INHAND_API_KEY=sk-...
# 可选：export INHAND_API_BASE=https://your-endpoint/v1 INHAND_MODEL=gpt-6
python scripts/run_sim.py --policy gpt --seed 0 --max-cycles 25 --video runs/gpt0.mp4 --export runs/gpt0.json
```

## 验收结果（2026-09-22 实测）

- `pip install -e .` 成功；`python -m inhand_gpt.policy` 自测 7/7 通过；
- 基元实测：roll_cw/roll_ccw 等在种子 0–4 的偏航/漂移数据见 DESIGN.md 第 3 节（方向正确性依赖接触几何，已如实记录）；
- 规则基线 5 局（种子 0–4，25 周期）：**1/5 成功**（seed 3 收敛到 0.099 rad < 0.1 rad）；其余各局最终误差与失败原因分析见 DESIGN.md 第 5 节（不粉饰：s0 掉落、s1/s2 口袋吸引子停滞、s4 基元全弱）；
- `runs/` 下有 5 个 JSON + 5 个 MP4（480×480）；
- `run_sim.py --policy gpt --dry-run` 打印完整 prompt 后正常退出，不发请求；缺 `INHAND_API_KEY` 时报错退出（exit 2）。

## 已知限制与后续路线

1. **仿真手是 Shadow Dexterous Hand，不是真实的 Revo1Touch**：Gymnasium-Robotics 官方不提供 Revo1 模型，Shadow Hand 操控环境是官方可复现基准。后续可用 brainco-description 的 **Revo2 MJCF** 模型升级替换（决策层/观测契约不变，只需重调基元幅值）。
2. **开环基元存在"口袋吸引子"**（块 yaw≡+54° mod 90 处全局停滞，详见 DESIGN.md 失败分析）：需要更丰富的反馈（视觉/触觉）或带接触感知的基元才能稳定穿越。
3. **触觉观测入口**：官方环境变体 `HandManipulateBlockRotateZ_BooleanTouchSensors-v1` 与 `..._ContinuousTouchSensors-v1` 已随 gymnasium-robotics 安装，可直接替换环境 ID 做触觉条件决策研究。
4. 观测目前用仿真真值（块位姿），未做视觉估计；`render_mode="rgb_array"` 已通（录像即由此生成），可作为视觉观测研究入口。
5. GPT 策略本次未真实调用；prompt 契约与解析器已就位（DESIGN.md 第 6 节）。

## 纪事约定

**每次修改项目，都必须在 `CHANGES.md` 追加一条记录**（日期、改动、原因、实测数据）。该约定自 2026-09-22 项目初始化起生效。
