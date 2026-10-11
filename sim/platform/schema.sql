-- ============================================================================
-- ASCE-CoC 数据库 schema
--
-- 设计原则
--   1. 纯 DB 落库：不再有 CSV 认知状态落盘，所有结果只落 SQLite 一份。
--   2. 没有认知枚举列。认知只以自然语言存在于 agent_turn.thought_text 与
--      agent_turn.chain_steps_json，以及结构化摘要在 agent_chain_step。
--   3. 每轮每 agent 一条 agent_turn 宽行，把"看到什么 / 想了什么 / 做了什么"
--      三类信息聚合在一处。若把三类信息拆进不同的表、只用 (agent_id, tick)
--      隐式关联，分析时需要多路 JOIN，而且无法保证各表写的是同一轮同一状态。
--   4. 所有时间列统一 DATETIME 文本，由沙盒时钟推进。
-- ============================================================================

-- ---------------------------------------------------------------- 用户与画像
CREATE TABLE IF NOT EXISTS user (
    user_id        INTEGER PRIMARY KEY,
    user_name      TEXT,
    name           TEXT,
    bio            TEXT,
    persona        TEXT,
    age            TEXT,
    gender         TEXT,
    mbti           TEXT,
    country        TEXT,
    profession     TEXT,
    economic_status TEXT,
    education      TEXT,
    interested_topics TEXT,
    influence_score REAL DEFAULT 0,
    is_control     INTEGER DEFAULT 0,
    is_real        INTEGER DEFAULT 0,
    num_followings INTEGER DEFAULT 0,
    num_followers  INTEGER DEFAULT 0,
    created_at     DATETIME,
    step_number    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_user_control ON user(is_control);

-- ---------------------------------------------------------------- 内容
CREATE TABLE IF NOT EXISTS post (
    post_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER,
    content        TEXT NOT NULL,
    topic          TEXT,
    created_at     DATETIME,
    step_number    INTEGER DEFAULT 0,
    num_likes      INTEGER DEFAULT 0,
    num_dislikes   INTEGER DEFAULT 0,
    num_shares     INTEGER DEFAULT 0,
    num_comments   INTEGER DEFAULT 0,
    is_seed        INTEGER DEFAULT 0,
    source_post_id INTEGER,
    conversation_id TEXT,
    extra_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_post_user ON post(user_id);
CREATE INDEX IF NOT EXISTS idx_post_step ON post(step_number);
CREATE INDEX IF NOT EXISTS idx_post_time ON post(created_at);

CREATE TABLE IF NOT EXISTS comment (
    comment_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id            INTEGER NOT NULL,
    user_id            INTEGER,
    parent_comment_id  INTEGER,
    content            TEXT NOT NULL,
    created_at         DATETIME,
    step_number        INTEGER DEFAULT 0,
    num_likes          INTEGER DEFAULT 0,
    num_dislikes       INTEGER DEFAULT 0,
    num_replies        INTEGER DEFAULT 0,
    is_guidance        INTEGER DEFAULT 0,
    is_seed            INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_comment_post ON comment(post_id);
CREATE INDEX IF NOT EXISTS idx_comment_parent ON comment(parent_comment_id);
CREATE INDEX IF NOT EXISTS idx_comment_user ON comment(user_id);

-- ---------------------------------------------------------------- 互动关系
CREATE TABLE IF NOT EXISTS post_like (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    post_id      INTEGER NOT NULL,
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0,
    UNIQUE(user_id, post_id)
);

CREATE TABLE IF NOT EXISTS post_dislike (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    post_id      INTEGER NOT NULL,
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0,
    UNIQUE(user_id, post_id)
);

CREATE TABLE IF NOT EXISTS comment_reaction (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    comment_id   INTEGER NOT NULL,
    kind         TEXT NOT NULL,          -- like / dislike
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0,
    UNIQUE(user_id, comment_id, kind)
);

CREATE TABLE IF NOT EXISTS follow (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    follower_id  INTEGER NOT NULL,
    followee_id  INTEGER NOT NULL,
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0,
    UNIQUE(follower_id, followee_id)
);

CREATE TABLE IF NOT EXISTS comment_share (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    comment_id   INTEGER NOT NULL,
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0
);

-- ---------------------------------------------------------------- 推荐与曝光
CREATE TABLE IF NOT EXISTS rec (
    user_id      INTEGER NOT NULL,
    post_id      INTEGER NOT NULL,
    step_number  INTEGER DEFAULT 0,
    score        REAL DEFAULT 0,
    rank         INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, post_id, step_number)
);
CREATE INDEX IF NOT EXISTS idx_rec_step ON rec(step_number);

-- ---------------------------------------------------------------- 轨迹
CREATE TABLE IF NOT EXISTS trace (
    trace_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER,
    action       TEXT,
    info         TEXT,
    created_at   DATETIME,
    step_number  INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_trace_user_step ON trace(user_id, step_number);
CREATE INDEX IF NOT EXISTS idx_trace_action_step ON trace(action, step_number);

-- ============================================================================
-- 认知链核心表
-- ============================================================================

-- 每轮每 agent 一行：完整因果链（看到 / 想到 / 做到）
CREATE TABLE IF NOT EXISTS agent_turn (
    agent_id            INTEGER NOT NULL,
    tick                INTEGER NOT NULL,

    -- 本轮状态
    is_active           INTEGER DEFAULT 0,
    status              TEXT DEFAULT 'ok',        -- ok / failed / skipped
    error_text          TEXT,

    -- ① 看到了什么
    feed_post_ids       TEXT,                     -- 本轮推荐流帖子 ID 列表 JSON
    feed_trending_ids   TEXT,                     -- 热门评论 ID 列表 JSON
    interaction_ids     TEXT,                     -- 定向互动 ID 列表 JSON

    -- ② 思考了什么
    thought_text        TEXT,                     -- 两阶段合并的完整思考原文（论文可直接引用）
    stage1_text         TEXT,                     -- 第 1 阶段原文（情绪/情感/思维/立场）
    stage2_text         TEXT,                     -- 第 2 阶段原文（观点/意图/一致性）
    chain_steps_json    TEXT,                     -- 节点级结构化 JSON（含 missing）

    -- ③ 做了什么
    actions_json        TEXT,                     -- 全部工具调用及结果 JSON
    primary_action      TEXT,                     -- 主动作名
    primary_target_id   INTEGER,                  -- 主动作对象 ID
    reason              TEXT,                     -- 本轮行为理由（取意图或主动作摘要）

    -- 元信息
    profile             TEXT,                     -- 使用的认知链档位 max/pro/flash
    stage1_ms           INTEGER,
    stage2_ms           INTEGER,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    model               TEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (agent_id, tick)
);
CREATE INDEX IF NOT EXISTS idx_turn_tick ON agent_turn(tick);
CREATE INDEX IF NOT EXISTS idx_turn_active ON agent_turn(tick, is_active);

-- 认知链节点级明细（长表，方便做机理分析与分布统计）
CREATE TABLE IF NOT EXISTS agent_chain_step (
    agent_id     INTEGER NOT NULL,
    tick         INTEGER NOT NULL,
    stage        INTEGER NOT NULL,      -- 1 或 2
    dim          TEXT NOT NULL,         -- emotion/mood/thinking/stance/opinion/intention/integration
    dim_label    TEXT,
    node_key     TEXT NOT NULL,         -- E1/M2/T1/S3/O2/I1/Z1 ...
    node_label   TEXT,
    step_order   INTEGER DEFAULT 0,
    text         TEXT,                  -- 该节点的推理文本
    present      INTEGER DEFAULT 1,     -- 该节点是否有内容
    inherited    INTEGER DEFAULT 0,     -- 1 = 内容由维度级兜底填充（模型没写节点名）
    PRIMARY KEY (agent_id, tick, node_key)
);
CREATE INDEX IF NOT EXISTS idx_chain_dim ON agent_chain_step(dim, tick);
CREATE INDEX IF NOT EXISTS idx_chain_node ON agent_chain_step(node_key, tick);

-- 工具调用明细（一个 turn 可多次调用）
CREATE TABLE IF NOT EXISTS agent_tool_call (
    agent_id       INTEGER NOT NULL,
    tick           INTEGER NOT NULL,
    call_order     INTEGER NOT NULL,
    tool_name      TEXT NOT NULL,
    tool_cn        TEXT,
    arguments_json TEXT,
    result_json    TEXT,
    success        INTEGER DEFAULT 0,
    error_text     TEXT,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, tick, call_order)
);
CREATE INDEX IF NOT EXISTS idx_toolname ON agent_tool_call(tool_name, tick);

-- 定向互动（后来者：本轮某 agent 收到的干预消息）
CREATE TABLE IF NOT EXISTS agent_interaction (
    agent_id        INTEGER NOT NULL,
    tick            INTEGER NOT NULL,
    interaction_id  INTEGER NOT NULL,
    from_id         INTEGER,
    from_name       TEXT,
    post_id         INTEGER,
    content         TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent_id, tick, interaction_id)
);

-- ============================================================================
-- 干预（成对影子设计：公共世界保持无干预，干预内容只进入隔离分支）
-- ============================================================================
CREATE TABLE IF NOT EXISTS intervention (
    intervention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id     TEXT,
    tick            INTEGER,
    target_id       INTEGER,
    guidance_id     INTEGER,
    strategy        TEXT,
    persona         TEXT,
    post_id         INTEGER,
    comment_id      INTEGER,
    content         TEXT,
    created_at      DATETIME
);
CREATE INDEX IF NOT EXISTS idx_iv_tick ON intervention(campaign_id, tick);

CREATE TABLE IF NOT EXISTS campaign_target (
    campaign_id   TEXT NOT NULL,
    target_id     INTEGER NOT NULL,
    selection_tick INTEGER,
    paired_from_tick INTEGER,
    status        TEXT DEFAULT 'assigned',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, target_id)
);

-- ============================================================================
-- 运行元信息
-- ============================================================================
CREATE TABLE IF NOT EXISTS sim_meta (
    key    TEXT PRIMARY KEY,
    value  TEXT,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sim_tick_stat (
    tick                INTEGER PRIMARY KEY,
    active_count        INTEGER DEFAULT 0,
    post_count          INTEGER DEFAULT 0,
    comment_count       INTEGER DEFAULT 0,
    like_count          INTEGER DEFAULT 0,
    follow_count        INTEGER DEFAULT 0,
    tool_call_count     INTEGER DEFAULT 0,
    chain_missing_count INTEGER DEFAULT 0,
    stage1_ms_total     INTEGER DEFAULT 0,
    stage2_ms_total     INTEGER DEFAULT 0,
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    elapsed_sec         REAL DEFAULT 0,
    finished_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
