# CHANGES.md — 项目纪事

本项目纪事约定：**每次修改项目，都必须在本文件追加一条记录**（日期、改动、原因、实测数据链接）。该约定同时写入 README.md。

---

## 2026-09-22 — 项目初始化：InHand-GPT 仿真核心

### 项目性质
- 本项目为**自研核心**（realman-inhand / inhand-gpt 包），无上游基座改动。
- Gymnasium-Robotics（含 MuJoCo 的 Shadow Dexterous Hand 操控环境）作为**未修改依赖**使用（pip 安装，版本 gymnasium-robotics 1.4.2，mujoco 3.13.0）。
- 仿真手为 **Shadow Dexterous Hand**，而非用户真实的 Revo1Touch 灵巧手。原因：Gymnasium-Robotics 官方未提供 Revo1 模型；`HandManipulateBlockRotateZ-v1` 是官方、可复现的掌内操控基准。后续可用 brainco-description 的 Revo2 MJCF 模型升级替换（见 README「已知限制与后续路线」）。

### 本次实现的关键决策
- **决策循环层级**：GPT-as-Policy 风格 —— 决策层每周期输出一个"基元选择"，基元为 12–30 个 env 步的开环相对关节增量序列。决策频率远低于控制频率，与真机 VLM 延迟相匹配。
- **动作语义**：环境 `relative_control=True`，action ∈ [-1,1]²⁰ 表示相对关节增量（约等于归一化关节速度）。因此"正弦波形基元"≈ 关节速度波形；"常值 bias"会持续爬升至关节限位（深屈拖拽类基元利用此机制，并已量化其漂移代价）。
- **基元演化**（全程由 `scripts/tune_primitives.py` 实测驱动，数据见 DESIGN.md）：
  1. 初版纯 bias 拖拽波形在闭环中暴露**关节饱和停滞**：1–3 次执行后关节到限位即失效（roll_cw 停滞于块 yaw≡+54°(mod 90)，两个种子独立复现）；
  2. v2 复合"拖拽+开合尾"反而把关节送回限位且个别种子掉块，弃用；
  3. v3 定版**复合波形**（拖拽段 + 关节回中段），可重复执行；删除实测掉块的 roll_mf（单次执行漂移 3.5–4.7cm）；
  4. 增补 roll_wr（腕部 WRJ1 摇摆，用于口袋逃逸）与 regrasp（接触重置）；实测 regrasp **不能**平移回中方块（位移卡在指墙间），如实记录。
- **决策策略演化**：符号规则 → 违规降级阶梯 → 全菜单 bandit（探索代价过高，一次反向探索 −17.9°）→ 定版 **sign-prior 候选对 + 在线助益度反馈 + regrasp 重置**（DESIGN.md 第 4 节）。
- **死区**：5°（死区内 `finish`，由 loop 按环境判据 rot_dist<0.1 rad≈5.73° 判成功）。
- **步数预算**：max-cycles 默认 25；env `max_episode_steps=2000`（放宽官方 100 步截断，避免 17 周期处被 TimeLimit 误杀）。
- **掉落判据**：水平漂移 >2.5cm 或 z 下降 >5cm（环境本身无掉落终止）。
- **观测契约**：决策层只拿结构化观测（带符号绕 Z 角误差 err_deg、几何旋转距离 rot_dist_rad、方块漂移、上一步动作与实测角位移、近期历史），不直接读图像。
- **GPT 策略**：OpenAI 兼容 `/chat/completions`（httpx 同步）；配置全走环境变量（INHAND_API_BASE / INHAND_API_KEY / INHAND_MODEL，默认模型 gpt-6）；**缺 key 立即报错退出（exit 2），绝不静默回退规则基线**；`--dry-run` 打印完整 prompt 不发请求；响应解析为可离线单测纯函数（`python -m inhand_gpt.policy` 自测 7/7 通过）。
- **工具链**：`pip install -e .` 可编辑安装（包名 inhand-gpt）；视频用 imageio-ffmpeg 编码 MP4，编码器不可用时退化 GIF 并警告；控制台输出统一英文（Windows/GBK 控制台中文会乱码），文档用中文。
- **bug 修复**：mujoco 3.x 的 MjData 无 `get_joint_qpos` 便捷方法（改为按 `jnt_qposadr` 直读）；`np.asarray` 不拷贝导致初始位姿视图随仿真更新、漂移恒为 0（改为 `np.array` 拷贝）。

### 验收结果（2026-09-22 实测，命令与数据均可复现）
1. `pip install -e .` 成功（inhand-gpt 0.1.0）。
2. 基元实测有效：`scripts/tune_primitives.py --scan pkg --seeds 0 1 2 3 4` 输出见 DESIGN.md 第 3 节（roll_cw −10~−15°/30步、roll_ccw +6~+10°/30步（方向随接触几何变化，已如实记录）、漂移 ≤1.5cm）。
3. 规则基线 5 局（种子 0–4，25 周期）：**成功率 1/5**——seed 3 收敛至 0.099 rad（<0.1 rad）判成功；s0 掉落（−44.1°）、s1 口袋极限环（+21.7°）、s2 口袋停滞（−76.1°）、s4 基元全弱（+129.1°）。失败分析见 DESIGN.md 第 5 节。
4. `runs/` 下 5 个 JSON + 5 个 MP4（480×480；seed1.mp4 741 帧，seed3.mp4 1 帧因第 0 周期即 finish）。
5. `run_sim.py --policy gpt --dry-run --seed 0 --max-cycles 1` 打印完整 prompt 后正常退出（未发请求）。
6. `python -m inhand_gpt.policy` 离线自测 7/7 通过（合法/围栏 JSON 解析、非 JSON 拒止、非法 choice 拒止、非对象拒止、缺 key 拒止、dry-run 免 key）。
7. README/DESIGN 中文；含 Shadow Hand 替代 Revo1 的原因、Revo2 MJCF 升级路径、触觉观测变体（`_BooleanTouchSensors`/`_ContinuousTouchSensors`）入口。

---

## 2026-09-23 — GPT 平台连通性测试：受阻（非代码改动）

- 用户提供中转平台 `https://apinebula.ai/v1` 及 API key（**key 不落入本仓库任何文件**，仅以环境变量传入进程）。
- 实测：`GET /v1/models` 该 key 仅返回 `gpt-image-2`、`gpt-image-2.5`（图像模型）；`gpt-6 / gpt-6-astra / gpt-6.1 / gpt-6-pro / gpt-5.x / gpt-4o / claude-sonnet-4.5 / deepseek-v3 / openai/*` 等全部聊天模型 ID 均返回 `get_channel_failed`（"当前服务暂不支持该模型"）。
- 结论：**该 key/账户当前无聊天补全模型权限**（疑似仅图像模型通道），GPT 决策实验阻塞，待用户在平台侧确认/开通聊天模型或更换平台后重试。

---

## 2026-09-23 — GPT-6-astra 对照实验（5 种子，真实 API）

- 用户更换 key 后实测：`/v1/models` 含 `gpt-6-astra`（另有 gpt-5.5、gpt-5.6-sol 等 4 个），chat 调用正常（usage 正常返回）。key 仍以环境变量传入，不落盘。
- 实验设置：与 2026-09-22 规则基线完全同种子（0–4）、同 25 周期预算、同基元菜单；决策模型 `gpt-6-astra`（经 `https://apinebula.ai/v1`，OpenAI 兼容协议）；共 101 次真实调用，平均 ~6.5s/次，单局最长 195s。
- **结果：GPT 1/5 成功，规则基线 1/5 成功**（同种子对照，逐周期数据与视频在 `runs/gpt_seed{0-4}.{json,mp4}`）：

| seed | 基线最终误差 | GPT 最终误差 | 备注 |
| --- | --- | --- | --- |
| 0 | −44.1°（掉落） | −42.4° | GPT 全程未掉落（基线唯一掉落局）；推进幅度相当 |
| 1 | +21.7° | +77.6° | 基线 sign-prior 蒙对有效基元大幅领先；GPT 在 5 种基元间振荡 |
| 2 | −76.1° | −73.2° | GPT 锁定有效基元 roll_lf×18 仍困于口袋；两者相当 |
| 3 | 成功（初始达标） | **成功**（1 周期 finish，16s） | GPT 正确识别已达目标，sanity check 通过 |
| 4 | +129.1° | +119.3° | 均基本无效 |

- 定性发现（rationale 原文存于导出 JSON）：GPT 的决策展现了真实的因果反馈推理（追踪各基元历史增益、监控漂移、增益衰退时主动换基元）；失败原因不是决策质量，而是**观测中无接触状态信息**——两个策略都倒在没有触觉的同一道墙上。这是触觉包装研究假设的第一次直接证据。
- 已知局限：单局样本量小（5 种子 × 1 局）；菜单文本中的实测数据（含方向不稳定声明）可能干扰 GPT 先验；未做温度/采样重复性控制。

---

## 2026-09-23 — 首次发布：上传 GitHub

- 项目以 **私有仓库**发布至 <https://github.com/OohLeeGen/InHand-GPT>（账户认证走本机 Windows 凭据管理器，token 不落盘不显示；网络经 `127.0.0.1:7897` 代理，已在仓库内设 `http.proxy` 便于后续 push/pull）。
- 私有为保守默认，可随时在仓库 Settings → General → Danger Zone 改为 Public。
- 首次提交内容：全部源码（`inhand_gpt/`）、脚本（`scripts/`）、文档（README/DESIGN/CHANGES）、实验产物（`runs/`：基线与 GPT 各 5 局的 JSON+MP4、基线迭代留档 JSON）、仿真可用性检查脚本（`sim/`）。
- 安全检查：发布前全仓扫描确认无 API key 等凭据（key 仅以环境变量传入进程，从未落盘）。
