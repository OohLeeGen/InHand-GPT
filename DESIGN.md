# DESIGN.md — InHand-GPT 设计文档（简版）

## 1. 系统分层

```
┌────────────────────────────────────────────────────────────┐
│ 决策层（低频，每周期 1 次）                                  │
│   RuleBaselinePolicy（规则基线，已验证）                     │
│   GPTPolicy（OpenAI 兼容 API，本次仅 dry-run，不实调）        │
│   输入：结构化观测 + 近期历史 + 基元菜单                      │
│   输出：{"choice": <基元名|finish>, "rationale": ...}        │
├────────────────────────────────────────────────────────────┤
│ 基元层（中频，每个基元 12–30 个 env 步的开环动作序列）        │
│   hold / roll_cw / roll_ccw / roll_lf / roll_th / roll_wr   │
│   regrasp / nudge_cw / nudge_ccw                            │
├────────────────────────────────────────────────────────────┤
│ 仿真层（高频，env.step）                                     │
│   Gymnasium-Robotics HandManipulateBlockRotateZ-v1          │
│   Shadow Dexterous Hand + 方块，MuJoCo 3.13                 │
└────────────────────────────────────────────────────────────┘
```

## 2. 观测契约（observe.py）

决策层只拿结构化观测，不直接读图像：

| 字段 | 含义 |
|---|---|
| `err_deg` | 带符号绕世界 +Z 角度误差（度）：`q_rel = q_ach * conj(q_des)`，`err = wrap(2*atan2(q_rel.z, q_rel.w))`。**err>0 表示方块需要绕 -Z 方向旋转**（err = 块 − 目标） |
| `rot_dist_rad` | 环境同款无符号测地旋转距离 `2*acos(|<q_ach,q_des>|)`，成功判据 <0.1 rad（≈5.73°），与 Gymnasium-Robotics 官方 `_is_success` 一致 |
| `block_pos` / `drift_xy_m` / `drift_z_m` | 方块中心位置与相对局初的漂移；漂移 >2.5cm 或 z 下降 >5cm 判定掉落失败 |
| `success` / `dropped` | 由 loop 按上述判据计算 |

执行器契约（已由源码与运行时核实）：动作空间 `Box(-1,1,(20,))`，`relative_control=True`，每步 `ctrl = 当前关节角 + action * (ctrlrange 半宽)`——即 action 是归一化**相对关节增量**（近似关节速度）。20 个执行器顺序：`WRJ1, WRJ0, FFJ3, FFJ2, FFJ1, MFJ3, MFJ2, MFJ1, RFJ3, RFJ2, RFJ1, LFJ4, LFJ3, LFJ2, LFJ1, THJ4, THJ3, THJ2, THJ1, THJ0`（FF=食指 MF=中指 RF=无名指 LF=小指 TH=拇指 WRJ=腕）。

## 3. 基元定义与实测数据（primitives.py）

设计要点：
- **复合波形**：纯"深屈拖拽"（恒值 bias）会在 1–3 次执行后把关节推到限位并完全失效（闭环实测：roll_cw 在块 yaw≡+54°(mod 90) 处饱和停滞）。因此主基元 = 拖拽段（bias+正弦波）+ **关节回中段**（把拖拽关节送回中位），可重复执行不失效。
- **零净位移波形**（roll_lf / roll_th / roll_wr / regrasp 的驱动关节走零均值正弦）天然可持续，但每次旋转量较小。
- 笼持关节（cage bias）推到限位后作为静态支架，不影响驱动关节的持续能力。

实测表（`scripts/tune_primitives.py --scan pkg`，种子 0–4，从 reset 单次执行；数值 = 带符号偏航°/最大漂移 cm；roll 系 30 步，nudge 12 步，hold/regrasp 20 步）：

| 基元 | seed0 | seed1 | seed2 | seed3 | seed4 | 方向先验 |
|---|---|---|---|---|---|---|
| hold | -3.3/0.2 | -0.2/0.1 | -1.1/0.1 | -0.4/0.1 | +2.5/0.1 | 无（被动漂移） |
| roll_cw | -15.0/0.8 | -11.9/1.3 | -10.1/1.5 | +0.3/1.5 | +0.9/0.4 | **负偏航**（s0–2 强，s3–4 弱） |
| roll_ccw | +6.3/1.1 | -2.8/0.9 | -3.4/0.9 | +10.1/0.3 | +2.2/0.8 | **正偏航**（s0/s3，s1–2 反号） |
| roll_lf | -1.6/0.9 | +15.7/1.0 | +16.1/1.1 | +14.5/0.6 | -0.0/0.4 | 正偏航（s1–3 强） |
| roll_th | -9.8/0.8 | -6.4/0.5 | -7.0/0.6 | +1.6/0.9 | +3.5/0.3 | 负偏航（s0–2） |
| roll_wr | -13.7/1.8 | -13.2/1.4 | -13.5/1.4 | +4.0/1.7 | -2.2/1.4 | 负偏航（s0–2），口袋逃逸用 |
| regrasp | -1.9/1.2 | -2.6/1.0 | -4.7/1.1 | +9.3/1.1 | +2.0/0.5 | 接触模式重置 |
| nudge_cw | -2.9/0.1 | -4.5/0.7 | -4.8/0.8 | +6.6/0.6 | +2.2/0.2 | 弱负偏航 |
| nudge_ccw | +1.5/0.9 | -1.1/0.5 | -1.4/0.5 | +4.5/0.3 | +1.8/0.4 | 弱正偏航 |

**关键事实：符号图随接触几何（种子）变化**——没有任何开环波形能在所有种子上保证同一方向。这是本任务的核心工程难点，决策层必须用实测反馈在线选择基元（规则基线：sign-prior 候选对 + 在线"助益度"反馈 + regrasp 重置；GPT：在 prompt 中给出上一步动作与实测角位移，由模型自行适配）。

## 4. 规则基线（RuleBaselinePolicy）

- 死区 5°：|err| ≤ 5° → `finish`，由 loop 按环境判据（rot_dist < 0.1 rad ≈ 5.73°）判成功。
- 候选对（先验方向匹配，避免反向探索的一次 −18° 级灾难）：
  - err>0（需负偏航）：`[roll_cw, roll_th, roll_wr]`（|err|<18° 时追加 `nudge_cw`）
  - err<0（需正偏航）：`[roll_ccw, roll_lf, roll_wr]`（|err|<18° 时追加 `nudge_ccw`）
- 在线反馈：每个候选的"助益度" = 最近 2 次使用（自上次 regrasp 起）的 `|err前|−|err后|` 均值，≥0.3° 则继续利用；否则探索候选对中使用次数 <2 的基元；全部停滞则 `regrasp` 重置接触模式。

## 5. 闭环 5 局实测（2026-09-22，max-cycles 25，种子 0–4）

| seed | 初始 err | 最终 err | 结果 | 备注 |
|---|---|---|---|---|
| 0 | -61.2° | -44.1° | 失败（掉落） | roll_ccw 每周期 +6~11° 但位移 ~1.1cm/周期，第 3 周期漂移 >2.5cm 判掉落 |
| 1 | +75.8° | +21.7° | 失败（周期用尽） | roll_cw 9 周期推进 −43°，随后进入 +21.5/+26.5 双吸引子极限环 |
| 2 | -116.6° | -76.1° | 失败（周期用尽） | roll_lf 持续 +1.5~7.5°/周期推进 −45°，在口袋角停滞 |
| 3 | -1.5° | -1.5° | **成功** | 初始即在 0.1 rad 判据内（rot_dist=0.099），第 0 周期 finish |
| 4 | +128.4° | +129.1° | 失败（周期用尽） | 该接触几何下所有候选基元均弱（<2°/周期） |

成功率 1/5。**失败分析（如实记录）**：
1. **口袋吸引子**：块体 yaw ≡ +54° (mod 90°) 处是全局动力学陷阱（立方体对角线卡入指缝），所有开环波形在该处衰减至零净旋转；s1/s2 均在此停滞。
2. **双吸引子极限环**（s1）：roll_th 把块从 +21.5 推回 +26.5，roll_wr 从 +26.5 拉回 +21.5，振荡无法穿越。
3. **拖拽-位移耦合**（s0）：强正偏航基元 roll_ccw 同时把块拖向指根，位移每周期 ~1.1cm，3 周期后触发 2.5cm 掉落判据；regrasp 实测**不能**使块回中（平移被困在指墙间）。
4. 逃逸口袋需要的可能是"抬升-旋转"或基于接触反馈的戳刺，这超出无传感开环波形的能力——正是引入视觉/触觉观测与 GPT 决策的研究动机（见 README 后续路线）。

## 6. Prompt 契约（GPTPolicy）

每轮发送：system（固定角色）+ user（`PROMPT_TEMPLATE`：观测语义说明 + 当前观测 JSON + 最近 5 周期历史 JSON + 基元菜单 JSON）。要求模型只返回：

```json
{"choice": "<基元名或 finish>", "rationale": "<一句话>"}
```

解析（`parse_gpt_response`，纯函数）：接受裸 JSON 或 ``` 围栏 JSON；拒绝非 JSON、非对象、choice 不在合法集合的响应（抛 `PolicyError`，由调用方决定重试或终止）。离线自测：`python -m inhand_gpt.policy`（7/7）。

配置全走环境变量：`INHAND_API_BASE`（默认 `https://api.openai.com/v1`）、`INHAND_API_KEY`（缺失立即报错，**绝不静默回退规则基线**）、`INHAND_MODEL`（默认 `gpt-6`）。`--dry-run` 打印完整 prompt 不发请求。httpx 同步 POST `{api_base}/chat/completions`，`temperature=0`，`response_format=json_object`。

## 7. 与真实 Revo1Touch 的对应关系

| 仿真（本项目） | 真机（后续迁移） |
|---|---|
| Shadow Dexterous Hand（官方模型；官方无 Revo1 模型，见 README） | Revo1Touch 灵巧手（或先用 brainco-description 的 Revo2 MJCF 在仿真中升级） |
| 相对关节增量位置控制（20 维） | Revo 位置/速度指令（SDK 关节角设定） |
| 块四元数真值（`object:joint` qpos） | 视觉/动捕估计的物块位姿 |
| 结构化观测 err_deg/drift | 由位姿估计计算；可加触觉（官方 `_BooleanTouchSensors`/`_ContinuousTouchSensors` 环境变体） |
| 基元 = 开环增量序列 | 同名基元可直接在真机重放（需重调幅值/步数），决策层与 prompt 契约不变 |
| GPT 决策每周期 1 次（~30 仿真步） | 与 VLM/LLM 实际延迟匹配的异步决策频率 |
