# Tg_translate_bot

Telegram 双语群聊自动翻译机器人。群里有人发中文就自动回一条越南语（或英语、日语……），发越南语就自动回中文。带上下文、术语表、按群设置语言对，自部署，单文件。

## 功能

- **自动双向翻译**：检测消息语言，属于本群语言对中的一种就翻译成另一种，以斜体静默回复（不打扰通知）。
- **带上下文**：翻译时把最近的对话（默认 10 条 / 15 分钟内）和被回复的那条消息一起交给模型，代词、省略主语、"第二个"这类指代能正确还原。
- **按群设置语言对**：`/lang` 弹出中文按钮菜单，点两下选好；不同群可以是不同语言对，未设置的群用全局默认。
- **术语表**：`/term 小王 = Wang` 锁定人名、项目名、黑话的固定译法，按群生效。
- **主人限制**：设置 `OWNER_ID` 后，只有主人在场的群才工作，被拉进其他群会自动退出。
- **个人 opt-out**：`/off` 后自己的消息不再被翻译，`/on` 恢复。
- **OpenAI 兼容后端**：走 Chat Completions 格式，`OPENAI_BASE_URL` 可指向官方、中转站或自建服务。
- 翻译结果按（原文、目标语言、上下文指纹）缓存，重复内容不重复计费。

## 群内命令

| 命令 | 说明 |
|---|---|
| `/lang` | 按钮菜单设置本群语言对 |
| `/lang zh vi` | 直接用代码设置（中文 ⇄ 越南语） |
| `/lang -` | 恢复全局默认 |
| `/term` | 查看本群术语表 |
| `/term 小王 = Wang` | 添加/更新术语 |
| `/term 小王 -` | 删除术语 |
| `/off` / `/on` | 关闭 / 恢复对我消息的翻译 |
| `/status` | 查看本群状态 |
| `/help` | 帮助（末尾显示你的用户 ID） |

支持的语言代码：`zh` 中文、`en` 英语、`vi` 越南语、`ja` 日语、`ko` 韩语、`th` 泰语、`id` 印尼语、`ru` 俄语、`es` 西班牙语、`fr` 法语、`de` 德语、`pt` 葡萄牙语、`ar` 阿拉伯语。

## 部署

需要 Python 3.10+。

### 1. 创建机器人

1. 找 [@BotFather](https://t.me/BotFather) 发 `/newbot`，拿到 token。
2. **发 `/setprivacy` 选择这个机器人 → Disable**。不关隐私模式，机器人在群里收不到普通消息，翻译功能不会工作。如果机器人在关闭隐私模式之前已经进了群，要踢出后重新拉进来才生效。

### 2. 安装

```bash
git clone <本仓库> /root/Tg_translate_bot
cd /root/Tg_translate_bot
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
chmod 600 .env
```

编辑 `.env`，至少填 `BOT_TOKEN` 和 `OPENAI_API_KEY`；用中转站的话改 `OPENAI_BASE_URL` 和 `MODEL`。所有配置项的说明见 `.env.example`。

### 3. 试运行

```bash
set -a; . ./.env; set +a
venv/bin/python bot.py
```

看到 `Application started` 就是连上了。把机器人拉进群，发 `/status` 验证。

### 4. 用 systemd 常驻

```bash
cp deploy/tgtranslate.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now tgtranslate
journalctl -u tgtranslate -f      # 看日志
```

单元文件默认假设项目在 `/root/Tg_translate_bot`，放在别处就改一下里面的路径。崩溃会在 5 秒后自动重启，开机自启。

改了代码或 `.env` 之后记得 `systemctl restart tgtranslate`。

### 5. 设置主人限制（可选）

和机器人私聊发 `/help`，末尾会显示你的用户 ID，填到 `.env` 的 `OWNER_ID`，重启即可。之后机器人只在有你的群里工作。

## 工作原理

1. 收到群消息 → 先存进该群的滚动历史（即使这条不翻译，它也是后面消息的上下文）。
2. 用 `langdetect` 判断语言，中日韩用字符集兜底（纯汉字判中文，有假名判日语，有谚文判韩语——`langdetect` 对短句不可靠）。
3. 语言属于本群语言对的一种 → 目标语言就是另一种；否则跳过。
4. 拼 prompt：术语表 + 最近对话 + 被回复的消息 + 待翻译消息，请求 `POST {OPENAI_BASE_URL}/chat/completions`。
5. 结果以斜体、静默、不带链接预览的方式回复到原消息下。

数据存储：术语表、每群语言对、opt-out 名单在 SQLite（`bot.db`）里，重启不丢；对话历史和翻译缓存在内存里，重启清空。

## 注意

- `.env` 和 `bot.db` 已在 `.gitignore` 里，不要提交。
- API 返回 4xx（key 错误、模型名不存在）不会重试，直接记 ERROR 日志；429/5xx 会退避重试 3 次。
- 越南语、英语等拉丁字母语言靠统计检测，一两个词的极短消息可能识别不出来而被跳过。
