use super::parse_utils::*;
use super::sqlite_ro::{open_sqlite_ro, virtual_path};
use super::{units_from_messages, AgentAdapter};
use crate::models::*;
use anyhow::{anyhow, Result};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Antigravity 适配器(Google):
/// - IDE 模式: 扫描 `~/.gemini/antigravity-ide/brain/<uuid>/.system_generated/logs/transcript.jsonl`
///   解析完整的用户请求、模型思考过程、工具调用与输出、以及检查点摘要。
/// - CLI 模式 (向后兼容): 若仅有 `~/.gemini/antigravity-cli/conversation_summaries.db`,
///   且会话在 IDE 中未出现，生成元数据级会话卡片。
pub struct AntigravityAdapter {
    brain_root: Option<PathBuf>,
    cli_db: Option<PathBuf>,
    projects_json: PathBuf,
    rows_cache: MtimeCache<Vec<AgRow>>,
    projects_cache: std::sync::Mutex<Option<(i64, HashMap<String, String>)>>,
}

impl AntigravityAdapter {
    pub fn new() -> Self {
        let home = super::home_dir().unwrap_or_default().join(".gemini");
        let ide_brain = home.join("antigravity-ide").join("brain");
        let brain_root = if !ide_brain.is_dir() && home.join("antigravity").join("brain").is_dir() {
            Some(home.join("antigravity").join("brain"))
        } else {
            Some(ide_brain)
        };
        let cli_db = Some(
            home.join("antigravity-cli")
                .join("conversation_summaries.db"),
        );
        let projects_json = home.join("projects.json");
        Self {
            brain_root,
            cli_db,
            projects_json,
            rows_cache: MtimeCache::new(),
            projects_cache: std::sync::Mutex::new(None),
        }
    }

    fn rows(&self) -> Option<Vec<AgRow>> {
        let db = self.cli_db.as_ref()?;
        let mtime = super::sqlite_ro::db_cache_stamp(db);
        self.rows_cache.get_or_try_build(mtime, || {
            let ro = open_sqlite_ro(db, "antigravity")?;
            let mut stmt = ro
                .conn
                .prepare(
                    "SELECT conversation_id, title, preview, step_count, last_modified_time, workspace_uris
                     FROM conversation_summaries
                     WHERE parent_conversation_id = '' AND nesting_depth = 0",
                )
                .ok()?;
            let rows = stmt
                .query_map([], |r| {
                    Ok(AgRow {
                        id: r.get(0)?,
                        title: r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        preview: r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                        step_count: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                        modified_ms: sqlite_dt_ms(r.get::<_, Option<String>>(4)?.unwrap_or_default().trim()),
                        cwd: first_workspace(&r.get::<_, Option<String>>(5)?.unwrap_or_default()),
                    })
                })
                .ok()?
                .collect::<rusqlite::Result<Vec<_>>>()
                .ok()?;
            Some(rows)
        })
    }

    fn build_cli_meta(&self, r: &SessionFileRef, row: &AgRow) -> SessionMeta {
        let title = Some(clean_title_candidate(&row.title))
            .filter(|t| !t.is_empty())
            .or_else(|| Some(clean_title_candidate(&row.preview)).filter(|t| !t.is_empty()))
            .unwrap_or_else(|| UNTITLED.to_string());
        let ts = if row.modified_ms > 0 {
            row.modified_ms
        } else {
            r.mtime_ms
        };
        SessionMeta {
            key: format!("antigravity:{}", row.id),
            host: String::new(),
            id: row.id.clone(),
            agent: AgentId::Antigravity,
            title,
            project_path: row.cwd.clone(),
            project_name: project_name_of(&row.cwd),
            file_path: r.file_path.clone(),
            created_at: ts,
            updated_at: ts,
            message_count: row.step_count,
            size_bytes: r.size,
            git_branch: None,
            model: None,
            tokens_used: None,
            archived: false,
            source: None,
            favorite: false,
            pinned: false,
        }
    }

    fn parse_cli(&self, r: &SessionFileRef) -> Result<(SessionMeta, Vec<TranscriptMessage>)> {
        let rows = self
            .rows()
            .ok_or_else(|| anyhow!("cannot open antigravity summaries store"))?;
        let row = rows
            .iter()
            .find(|x| x.id == r.native_id)
            .ok_or_else(|| anyhow!("antigravity conversation {} not in store", r.native_id))?;

        let mut text = String::new();
        if !row.preview.trim().is_empty() {
            text.push_str(row.preview.trim());
            text.push_str("\n\n");
        }
        text.push_str("Antigravity stores conversation content encrypted — only this summary is available in Wake.");
        let mut messages = vec![text_msg(Role::System, &text, row.modified_ms)];
        assign_seq(&mut messages);
        Ok((self.build_cli_meta(r, row), messages))
    }

    fn projects_map(&self) -> HashMap<String, String> {
        let mtime = std::fs::metadata(&self.projects_json)
            .map(|m| mtime_ms(&m))
            .unwrap_or(0);
        {
            let cache = self.projects_cache.lock().unwrap();
            if let Some((t, map)) = cache.as_ref() {
                if *t == mtime {
                    return map.clone();
                }
            }
        }
        let mut out = HashMap::new();
        if let Ok(raw) = std::fs::read_to_string(&self.projects_json) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(serde_json::Value::Object(map)) = v.get("projects") {
                    for (path, name) in map {
                        if let Some(n) = name.as_str() {
                            out.insert(path.clone(), n.to_string());
                        }
                    }
                }
            }
        }
        *self.projects_cache.lock().unwrap() = Some((mtime, out.clone()));
        out
    }

    fn lookup_project_path(&self, candidate: &str) -> Option<String> {
        if candidate.is_empty() {
            return None;
        }
        let map = self.projects_map();
        let cand_norm = candidate.to_lowercase().replace('/', "\\");
        let mut best: Option<(&String, usize)> = None;
        for (proj_path, _) in &map {
            let p_norm = proj_path.to_lowercase().replace('/', "\\");
            if cand_norm.starts_with(&p_norm) && best.as_ref().map_or(true, |(_, len)| p_norm.len() > *len) {
                best = Some((proj_path, p_norm.len()));
            }
        }
        best.map(|(p, _)| p.clone())
    }

    fn build_ide_meta(&self, r: &SessionFileRef, parsed: &AntigravityIdeParse) -> SessionMeta {
        let title = parsed
            .title
            .as_deref()
            .filter(|t| !t.is_empty())
            .map(clean_title_candidate)
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| UNTITLED.to_string());

        let mut project_path = parsed.project_path.clone().unwrap_or_default();
        if !project_path.is_empty() {
            let p = Path::new(&project_path);
            if p.is_file() || p.extension().is_some() {
                if let Some(parent) = p.parent() {
                    project_path = parent.to_string_lossy().to_string();
                }
            }
        }
        if let Some(mapped) = self.lookup_project_path(&project_path) {
            project_path = mapped;
        }

        let created_at = if parsed.created_at > 0 {
            parsed.created_at
        } else {
            r.mtime_ms
        };
        let updated_at = if parsed.updated_at > 0 {
            parsed.updated_at
        } else {
            r.mtime_ms
        };

        SessionMeta {
            key: format!("antigravity:{}", r.native_id),
            host: String::new(),
            id: r.native_id.clone(),
            agent: AgentId::Antigravity,
            title,
            project_name: project_name_of(&project_path),
            project_path,
            file_path: r.file_path.clone(),
            created_at,
            updated_at,
            message_count: parsed.messages.len() as i64,
            size_bytes: r.size,
            git_branch: None,
            model: parsed.model.clone(),
            tokens_used: None,
            archived: false,
            source: None,
            favorite: false,
            pinned: false,
        }
    }

    fn parse(&self, r: &SessionFileRef, decode_images: bool) -> Result<(SessionMeta, Vec<TranscriptMessage>, u32)> {
        if r.file_path.contains('#') {
            let (meta, messages) = self.parse_cli(r)?;
            Ok((meta, messages, 0))
        } else {
            let parsed = parse_antigravity_jsonl(Path::new(&r.file_path), decode_images)?;
            let meta = self.build_ide_meta(r, &parsed);
            Ok((meta, parsed.messages, parsed.unknown_lines))
        }
    }
}

#[derive(Clone)]
struct AgRow {
    id: String,
    title: String,
    preview: String,
    step_count: i64,
    modified_ms: i64,
    cwd: String,
}

struct AntigravityIdeParse {
    messages: Vec<TranscriptMessage>,
    title: Option<String>,
    project_path: Option<String>,
    model: Option<String>,
    created_at: i64,
    updated_at: i64,
    unknown_lines: u32,
}

#[derive(Default)]
struct PendingAssistant {
    content: String,
    thinking: Option<String>,
    tool_calls: Vec<ToolCallView>,
    timestamp: Option<i64>,
    model: Option<String>,
}

fn flush_pending_assistant(
    pending: &mut PendingAssistant,
    messages: &mut Vec<TranscriptMessage>,
) {
    if pending.content.is_empty()
        && pending.thinking.is_none()
        && pending.tool_calls.is_empty()
    {
        return;
    }
    let (clipped, truncated) = clip(&pending.content, MAX_MSG_TEXT);
    let thinking = pending
        .thinking
        .take()
        .map(|t| clip(&t, MAX_TOOL_IO).0);
    messages.push(TranscriptMessage {
        seq: 0,
        role: Role::Assistant,
        kind: MessageKind::Text,
        text: clipped,
        truncated,
        tool_calls: std::mem::take(&mut pending.tool_calls),
        thinking,
        timestamp: pending.timestamp,
        model: pending.model.take(),
        images: Vec::new(),
    });
    pending.content.clear();
    pending.timestamp = None;
}

fn extract_user_request(raw: &str) -> (String, Option<String>, Option<String>) {
    let user_req = if let Some(start) = raw.find("<USER_REQUEST>") {
        let after = &raw[start + "<USER_REQUEST>".len()..];
        if let Some(end) = after.find("</USER_REQUEST>") {
            after[..end].trim().to_string()
        } else {
            after.trim().to_string()
        }
    } else {
        raw.trim().to_string()
    };

    let mut doc_path = None;
    if let Some(pos) = raw.find("Active Document:") {
        let after = &raw[pos + "Active Document:".len()..];
        let line = after.lines().next().unwrap_or("").trim();
        let path_part = if let Some((p, _)) = line.split_once(" (") {
            p.trim()
        } else {
            line
        };
        if !path_part.is_empty() {
            doc_path = Some(path_part.to_string());
        }
    }

    let mut model = None;
    if let Some(pos) = raw.find("Model Selection` from") {
        let after = &raw[pos..];
        if let Some(to_pos) = after.find(" to ") {
            let after_to = &after[to_pos + 4..];
            let end_idx = after_to
                .find(". No need")
                .or_else(|| after_to.find(".\r\n"))
                .or_else(|| after_to.find(".\n"))
                .or_else(|| after_to.find('\n'))
                .unwrap_or(after_to.len());
            let m = after_to[..end_idx].trim().trim_end_matches('.');
            if !m.is_empty() {
                model = Some(m.to_string());
            }
        }
    }

    (user_req, doc_path, model)
}

fn parse_tool_args(
    _tool_name: &str,
    args_val: Option<&serde_json::Value>,
) -> (String, Option<String>, Option<String>) {
    let Some(args_val) = args_val else {
        return (String::new(), None, None);
    };

    let obj: Option<serde_json::Value> = match args_val {
        serde_json::Value::Object(_) => Some(args_val.clone()),
        serde_json::Value::String(s) => serde_json::from_str(s).ok(),
        _ => None,
    };

    let mut preview = String::new();
    let mut cwd = None;

    if let Some(serde_json::Value::Object(map)) = obj.as_ref() {
        for key in &["Cwd", "DirectoryPath", "SearchPath"] {
            if let Some(val) = map.get(*key).and_then(|v| v.as_str()) {
                let v = val.trim_matches(|c| c == '"' || c == '\'').trim();
                if !v.is_empty() {
                    cwd = Some(v.to_string());
                    break;
                }
            }
        }

        for key in &[
            "toolSummary",
            "toolAction",
            "CommandLine",
            "TargetFile",
            "AbsolutePath",
            "Query",
            "DirectoryPath",
        ] {
            if let Some(val) = map.get(*key).and_then(|v| v.as_str()) {
                let v = val.trim_matches(|c| c == '"' || c == '\'').trim();
                if !v.is_empty() {
                    preview = v.to_string();
                    break;
                }
            }
        }
    }

    if preview.is_empty() {
        preview = match args_val {
            serde_json::Value::String(s) => s.trim().to_string(),
            _ => args_val.to_string(),
        };
    }
    let preview = clip(&preview, 120).0;

    let input_str = match args_val {
        serde_json::Value::String(s) => Some(s.clone()),
        _ => Some(args_val.to_string()),
    };

    (preview, input_str, cwd)
}

fn parse_antigravity_jsonl(path: &Path, _decode_images: bool) -> Result<AntigravityIdeParse> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(1 << 20, file);

    let mut messages: Vec<TranscriptMessage> = Vec::new();
    let mut unknown_lines = 0u32;
    let mut first_ts = 0i64;
    let mut last_ts = 0i64;
    let mut title_candidate = None;
    let mut candidate_doc_path = None;
    let mut candidate_tool_cwd = None;
    let mut current_model = None;

    let mut pending_assistant = PendingAssistant::default();

    for line in reader.lines() {
        let Ok(line) = line else {
            unknown_lines += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(row) = serde_json::from_str::<serde_json::Value>(&line) else {
            unknown_lines += 1;
            continue;
        };

        let step_type = row.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let ts_str = row.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let ts = iso_ms(ts_str);
        if ts > 0 {
            if first_ts == 0 {
                first_ts = ts;
            }
            last_ts = ts;
        }

        match step_type {
            "USER_INPUT" => {
                flush_pending_assistant(&mut pending_assistant, &mut messages);
                let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let (user_req, doc_path, model_opt) = extract_user_request(content);
                if title_candidate.is_none() && !user_req.is_empty() {
                    let cleaned = clean_title_candidate(&user_req);
                    if !cleaned.is_empty() {
                        title_candidate = Some(cleaned);
                    }
                }
                if candidate_doc_path.is_none() && doc_path.is_some() {
                    candidate_doc_path = doc_path;
                }
                if let Some(m) = model_opt {
                    current_model = Some(m);
                }

                let (clipped, truncated) = clip(&user_req, MAX_MSG_TEXT);
                messages.push(TranscriptMessage {
                    seq: 0,
                    role: Role::User,
                    kind: user_kind(&user_req),
                    text: clipped,
                    truncated,
                    tool_calls: Vec::new(),
                    thinking: None,
                    timestamp: if ts > 0 { Some(ts) } else { None },
                    model: current_model.clone(),
                    images: Vec::new(),
                });
            }
            "PLANNER_RESPONSE" => {
                if pending_assistant.timestamp.is_none() && ts > 0 {
                    pending_assistant.timestamp = Some(ts);
                }
                if current_model.is_some() && pending_assistant.model.is_none() {
                    pending_assistant.model = current_model.clone();
                }
                if let Some(th) = row.get("thinking").and_then(|v| v.as_str()) {
                    if !th.trim().is_empty() {
                        if let Some(ref mut existing) = pending_assistant.thinking {
                            existing.push_str("\n\n");
                            existing.push_str(th.trim());
                        } else {
                            pending_assistant.thinking = Some(th.trim().to_string());
                        }
                    }
                }
                if let Some(c) = row.get("content").and_then(|v| v.as_str()) {
                    if !c.trim().is_empty() {
                        if !pending_assistant.content.is_empty() {
                            pending_assistant.content.push_str("\n\n");
                        }
                        pending_assistant.content.push_str(c.trim());
                    }
                }
                if let Some(calls) = row.get("tool_calls").and_then(|v| v.as_array()) {
                    let step_idx = row.get("step_index").and_then(|v| v.as_i64()).unwrap_or(0);
                    for (ci, tc) in calls.iter().enumerate() {
                        let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                        let args = tc.get("args");
                        let (preview, input_str, cwd_cand) = parse_tool_args(name, args);
                        if candidate_tool_cwd.is_none() && cwd_cand.is_some() {
                            candidate_tool_cwd = cwd_cand;
                        }
                        pending_assistant.tool_calls.push(ToolCallView {
                            id: format!("tc-{}-{}", step_idx, ci),
                            name: name.to_string(),
                            input_preview: preview,
                            input: input_str,
                            output: None,
                            is_error: false,
                            sidechain_ref: None,
                        });
                    }
                }
            }
            "VIEW_FILE" | "RUN_COMMAND" | "GREP_SEARCH" | "LIST_DIRECTORY" | "CODE_ACTION"
            | "BROWSER_SUBAGENT" | "SEARCH_WEB" | "READ_URL_CONTENT" | "ASK_QUESTION"
            | "GENERIC" | "ERROR_MESSAGE" => {
                let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
                let is_err = status == "ERROR" || step_type == "ERROR_MESSAGE";
                let mut matched = false;
                for tc in pending_assistant.tool_calls.iter_mut().rev() {
                    if tc.output.is_none() {
                        tc.output = Some(clip(content.trim(), MAX_TOOL_IO).0);
                        tc.is_error = is_err;
                        matched = true;
                        break;
                    }
                }
                if !matched && is_err && !content.trim().is_empty() {
                    let (clipped, truncated) = clip(content.trim(), MAX_MSG_TEXT);
                    messages.push(TranscriptMessage {
                        seq: 0,
                        role: Role::System,
                        kind: MessageKind::Text,
                        text: clipped,
                        truncated,
                        tool_calls: Vec::new(),
                        thinking: None,
                        timestamp: if ts > 0 { Some(ts) } else { None },
                        model: None,
                        images: Vec::new(),
                    });
                }
            }
            "CHECKPOINT" => {
                flush_pending_assistant(&mut pending_assistant, &mut messages);
                let content = row.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !content.trim().is_empty() {
                    let (clipped, truncated) = clip(content.trim(), MAX_MSG_TEXT);
                    messages.push(TranscriptMessage {
                        seq: 0,
                        role: Role::System,
                        kind: MessageKind::CompactSummary,
                        text: clipped,
                        truncated,
                        tool_calls: Vec::new(),
                        thinking: None,
                        timestamp: if ts > 0 { Some(ts) } else { None },
                        model: None,
                        images: Vec::new(),
                    });
                }
            }
            "CONVERSATION_HISTORY" | "KNOWLEDGE_ARTIFACTS" | "SYSTEM_MESSAGE" => {}
            _ => {
                unknown_lines += 1;
            }
        }
    }

    flush_pending_assistant(&mut pending_assistant, &mut messages);
    assign_seq(&mut messages);

    Ok(AntigravityIdeParse {
        messages,
        title: title_candidate,
        project_path: candidate_tool_cwd.or(candidate_doc_path),
        model: current_model,
        created_at: first_ts,
        updated_at: last_ts,
        unknown_lines,
    })
}

/// workspace_uris JSON 数组("[\"file:///Users/…\"]")首项 → 本地路径
fn first_workspace(raw: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return String::new();
    };
    let Some(uri) = v
        .as_array()
        .and_then(|a| a.first())
        .and_then(|x| x.as_str())
    else {
        return String::new();
    };
    let path = uri.strip_prefix("file://").unwrap_or(uri);
    percent_decode(path)
}

/// file:// URI 的最小 percent-decode(路径含空格/中文时是 %XX 编码)
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

impl AgentAdapter for AntigravityAdapter {
    fn agent(&self) -> AgentId {
        AgentId::Antigravity
    }

    fn list_session_files(&self) -> Result<Vec<SessionFileRef>> {
        let mut refs = Vec::new();
        let mut seen_ids = HashSet::new();

        // 1. 枚举 IDE brain 目录下的 transcript.jsonl
        if let Some(ref brain) = self.brain_root {
            if let Ok(entries) = std::fs::read_dir(brain) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_dir() {
                        continue;
                    }
                    let id = entry.file_name().to_string_lossy().to_string();
                    if id.starts_with('.') || id == "tempmediaStorage" {
                        continue;
                    }
                    let transcript = path.join(".system_generated").join("logs").join("transcript.jsonl");
                    if let Ok(meta) = std::fs::metadata(&transcript) {
                        if meta.is_file() && meta.len() > 0 {
                            refs.push(SessionFileRef {
                                agent: AgentId::Antigravity,
                                native_id: id.clone(),
                                file_path: transcript.to_string_lossy().to_string(),
                                mtime_ms: mtime_ms(&meta),
                                size: meta.len() as i64,
                            });
                            seen_ids.insert(id);
                        }
                    }
                }
            }
        }

        // 2. 枚举 CLI db
        if let Some(ref db) = self.cli_db {
            if let Some(rows) = self.rows() {
                for row in rows {
                    if seen_ids.contains(&row.id) {
                        continue;
                    }
                    refs.push(SessionFileRef {
                        agent: AgentId::Antigravity,
                        native_id: row.id.clone(),
                        file_path: virtual_path(db, &row.id),
                        mtime_ms: row.modified_ms,
                        size: (row.title.len() + row.preview.len()) as i64,
                    });
                }
            }
        }

        Ok(refs)
    }

    fn file_ref(&self, path: &Path) -> Option<SessionFileRef> {
        let name = path.file_name()?.to_str()?;
        if name != "transcript.jsonl" {
            return None;
        }
        let logs = path.parent()?;
        if logs.file_name()?.to_str()? != "logs" {
            return None;
        }
        let sys = logs.parent()?;
        if sys.file_name()?.to_str()? != ".system_generated" {
            return None;
        }
        let session_dir = sys.parent()?;
        let native_id = session_dir.file_name()?.to_str()?.to_string();
        if native_id.starts_with('.') || native_id == "tempmediaStorage" {
            return None;
        }
        let meta = std::fs::metadata(path).ok()?;
        if !meta.is_file() || meta.len() == 0 {
            return None;
        }
        Some(SessionFileRef {
            agent: AgentId::Antigravity,
            native_id,
            file_path: path.to_string_lossy().to_string(),
            mtime_ms: mtime_ms(&meta),
            size: meta.len() as i64,
        })
    }

    fn quick_meta(&self, refs: &[SessionFileRef]) -> Option<HashMap<String, SessionMeta>> {
        let rows = self.rows()?;
        let by_id: HashMap<&str, &AgRow> = rows.iter().map(|r| (r.id.as_str(), r)).collect();
        let mut out = HashMap::new();
        for r in refs {
            if r.file_path.contains('#') {
                if let Some(row) = by_id.get(r.native_id.as_str()) {
                    out.insert(r.file_path.clone(), self.build_cli_meta(r, row));
                }
            }
        }
        Some(out)
    }

    fn parse_session(&self, r: &SessionFileRef) -> Result<ParsedSession> {
        let (meta, messages, unknown_line_count) = self.parse(r, false)?;
        let units = units_from_messages(&messages);
        Ok(ParsedSession {
            meta,
            units,
            unknown_line_count,
        })
    }

    fn parse_transcript(&self, r: &SessionFileRef) -> Result<ParsedTranscript> {
        let (meta, messages, unknown_line_count) = self.parse(r, true)?;
        Ok(ParsedTranscript {
            meta,
            mainline: messages,
            sidechains: Vec::new(),
            unknown_line_count,
        })
    }

    fn session_paths(&self, meta: &SessionMeta) -> Vec<String> {
        let p = Path::new(&meta.file_path);
        if meta.file_path.ends_with("transcript.jsonl") {
            if let Some(session_dir) = p.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
                if session_dir.is_dir() {
                    return vec![session_dir.to_string_lossy().to_string()];
                }
            }
        }
        vec![meta.file_path.clone()]
    }

    fn with_custom_root(&self, dir: PathBuf) -> Box<dyn AgentAdapter> {
        let (brain_root, cli_db) = if dir.is_file() {
            (None, Some(dir.clone()))
        } else if dir.file_name().and_then(|s| s.to_str()) == Some("brain") {
            (Some(dir.clone()), None)
        } else if dir.join("brain").is_dir() {
            (Some(dir.join("brain")), None)
        } else if dir.join("antigravity-ide").join("brain").is_dir() {
            let b = dir.join("antigravity-ide").join("brain");
            let d = dir.join("antigravity-cli").join("conversation_summaries.db");
            (Some(b), if d.is_file() { Some(d) } else { None })
        } else if dir.join("conversation_summaries.db").is_file() {
            (None, Some(dir.join("conversation_summaries.db")))
        } else {
            let nested = dir.join("antigravity-cli").join("conversation_summaries.db");
            let db = if nested.is_file() {
                Some(nested)
            } else {
                Some(dir.join("conversation_summaries.db"))
            };
            (None, db)
        };

        let projects_json = dir.join("projects.json");
        let projects_json = if projects_json.is_file() {
            projects_json
        } else {
            super::home_dir().unwrap_or_default().join(".gemini").join("projects.json")
        };

        Box::new(Self {
            brain_root,
            cli_db,
            projects_json,
            rows_cache: MtimeCache::new(),
            projects_cache: std::sync::Mutex::new(None),
        })
    }

    fn data_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        if let Some(ref b) = self.brain_root {
            roots.push(b.clone());
        }
        if let Some(ref d) = self.cli_db {
            roots.push(d.clone());
        }
        roots
    }
}

