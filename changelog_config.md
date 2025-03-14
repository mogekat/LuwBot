# Changelog

## [0.0.5] - 2025-3-11
### Added
- 新增了 `alias_names` 配置项，用于指定麦麦的别名。

## [0.0.4] - 2025-3-9
### Added
- 新增了 `memory_ban_words` 配置项，用于指定不希望记忆的词汇。

## v0.0.8
- 新增：消息跟踪功能配置，使机器人能够在发送消息后跟踪后续对话并决定是否继续回复
  - `follow_up_enabled`：是否启用消息跟踪功能
  - `follow_up_timeout`：跟踪超时时间（秒）
  - `follow_up_max_messages`：跟踪最大消息数
  - `follow_up_llm_prompt`：判断是否需要回复的LLM提示
  - `follow_up_llm`：消息跟踪功能使用的LLM模型，可单独配置



