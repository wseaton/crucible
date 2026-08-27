//! Parse Claude Code's `--output-format stream-json` NDJSON into [`AgentEvent`]s.
//!
//! The parser is stateful: token totals accrue across messages, a `tool_use` block's JSON input
//! arrives in `input_json_delta` chunks and is only complete at `content_block_stop`, and
//! text/thinking deltas buffer to line boundaries so one event is one line. Lines with an
//! unmodeled top-level `type`, and non-JSON lines, pass through as [`AgentEvent::Raw`] so a
//! viewer never loses output to schema drift, and so a startup failure claude prints as bare
//! text reaches the turn's log instead of being dropped.

use crate::otel::{CostHandle, LiveMeters, RateHandle};
use crucible_contract::event::{AgentEvent, RawStream, Tokens};
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;

/// Emit a [`Tokens`] sample once the running total has grown by at least this much
/// (or on the first sample).
const TOKEN_EMIT_STEP: u64 = 5_000;

/// Byte bound on each verbose tool input / result excerpt, so one giant Write or
/// Read cannot balloon the session log.
pub(crate) const TOOL_IO_LIMIT: usize = 2_048;

/// Which streamed text block is currently open (its deltas buffer to line boundaries).
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextKind {
    Text,
    Thinking,
}

impl TextKind {
    fn event(self, delta: String) -> AgentEvent {
        match self {
            TextKind::Text => AgentEvent::Text { delta },
            TextKind::Thinking => AgentEvent::Thinking { delta },
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-copy deserialization helpers for the hot path
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CbStartEvent<'a> {
    #[serde(borrow)]
    content_block: CbInfo<'a>,
}

#[derive(Deserialize)]
struct CbInfo<'a> {
    #[serde(borrow, rename = "type")]
    kind: &'a str,
    #[serde(borrow, default)]
    name: Option<Cow<'a, str>>,
    #[serde(borrow, default)]
    id: Option<Cow<'a, str>>,
}

#[derive(Deserialize)]
struct MsgStartEvent {
    message: MsgUsageWrap,
}

#[derive(Deserialize)]
struct MsgUsageWrap {
    #[serde(default)]
    usage: UsageFields,
}

#[derive(Deserialize, Default)]
struct UsageFields {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[allow(dead_code)]
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Deserialize)]
struct MsgDeltaEvent {
    #[serde(default)]
    usage: Option<UsageDeltaFields>,
}

#[derive(Deserialize)]
struct UsageDeltaFields {
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Deserialize)]
struct ErrorEvt<'a> {
    #[serde(borrow)]
    error: ErrInfo<'a>,
}

#[derive(Deserialize)]
struct ErrInfo<'a> {
    #[serde(borrow, rename = "type", default)]
    error_type: Cow<'a, str>,
    #[serde(borrow, default)]
    message: Cow<'a, str>,
}

// ---------------------------------------------------------------------------

/// Stateful decoder: feed it one stdout line at a time via [`StreamJsonParser::push`].
#[derive(Default)]
pub struct StreamJsonParser {
    // Cumulative token counts: input/cache from `message_start`, output from `message_delta`.
    input: u64,
    output: u64,
    cache_read: u64,
    cache_write: u64,
    last_emitted_total: u64,

    // The open text/thinking block and its partial (sub-line) tail.
    block: Option<TextKind>,
    buf: String,

    // The in-progress `tool_use` block: name + accumulated `input_json_delta`.
    tool_name: Option<String>,
    tool_json: String,

    // Live token rate and running cost from the OTLP collector; `None` when no collector is
    // running.
    meters: Option<LiveMeters>,
    // The stream's own cumulative cost, which only arrives with the turn-end `result` message.
    // Kept so a live cost sample can never report LESS than the agent has already declared.
    stream_cost: f64,

    // Verbose tool IO (CRUCIBLE_SESSION_TOOL_IO=full): tool events carry bounded
    // inputs, and the tool results claude echoes back as `user` messages become
    // result-excerpt events. Off by default so the session log stays compact.
    tool_io: bool,
    // The open `tool_use` block's id, and id -> name for labeling result excerpts.
    // Only populated under verbose tool IO.
    tool_id: Option<String>,
    tool_names: HashMap<String, String>,
}

impl StreamJsonParser {
    /// A parser that stamps each `tokens` sample with the collector's live meters: the 60 s-window
    /// token rate and the turn's cost so far.
    pub fn with_meters(meters: LiveMeters) -> Self {
        StreamJsonParser {
            meters: Some(meters),
            ..Default::default()
        }
    }

    /// Opt into verbose tool IO: tool events carry bounded inputs and result excerpts.
    pub fn with_tool_io(mut self, on: bool) -> Self {
        self.tool_io = on;
        self
    }

    /// Decode one line of claude `stream-json`, returning every [`AgentEvent`] it completed.
    pub fn push(&mut self, line: &str) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        self.push_into(line, &mut out);
        out
    }

    /// Like [`push`](Self::push) but appends events to a caller-owned buffer,
    /// avoiding a per-line allocation.
    pub fn push_into(&mut self, line: &str, out: &mut Vec<AgentEvent>) {
        let line = line.trim();
        if line.is_empty() {
            return;
        }

        match quick_type(line) {
            Some("assistant") | Some("rate_limit_event") => {}
            Some("stream_event") => self.stream_event_fast(line, out),
            Some("system") => {
                if let Ok(msg) = serde_json::from_str::<Value>(line) {
                    self.system(&msg, out);
                }
            }
            Some("result") => {
                if let Ok(msg) = serde_json::from_str::<Value>(line) {
                    self.end_block(out);
                    let is_error = bool_field(&msg, "is_error");
                    let error = is_error
                        .then(|| {
                            let r = str_field(&msg, "result");
                            if r.is_empty() {
                                str_field(&msg, "error")
                            } else {
                                r
                            }
                        })
                        .filter(|s| !s.is_empty());
                    let cost_usd = f64_field(&msg, "total_cost_usd");
                    self.stream_cost = self.stream_cost.max(cost_usd);
                    out.push(AgentEvent::Result {
                        subtype: str_field(&msg, "subtype"),
                        is_error,
                        turns: u64_field(&msg, "num_turns") as u32,
                        cost_usd,
                        error,
                    });
                }
            }
            Some("user") => {
                if let Ok(msg) = serde_json::from_str::<Value>(line) {
                    self.tool_results(&msg, out);
                }
            }
            Some(_) => out.push(raw(line)),
            None => out.push(raw(line)),
        }
    }

    /// `system/init` -> [`AgentEvent::Init`]; `system/api_retry` -> [`AgentEvent::Retry`].
    fn system(&mut self, msg: &Value, out: &mut Vec<AgentEvent>) {
        match msg.get("subtype").and_then(Value::as_str) {
            Some("init") => out.push(AgentEvent::Init {
                model: str_field(msg, "model"),
                tools: array_len(msg, "tools"),
                agents: array_len(msg, "agents"),
            }),
            Some("api_retry") => out.push(AgentEvent::Retry {
                attempt: u64_field(msg, "attempt") as u32,
                max: u64_field(msg, "max_retries") as u32,
                error: retry_error(msg),
            }),
            _ => {}
        }
    }

    /// Fast-path dispatch for stream_event lines. For the hottest path
    /// (content_block_delta), extracts the delta directly from the line without
    /// isolating the event JSON first. For other event types, falls back to
    /// brace-tracking extraction + typed deserialization.
    fn stream_event_fast(&mut self, line: &str, out: &mut Vec<AgentEvent>) {
        let Some(inner_type) = inner_event_type(line) else {
            return;
        };

        match inner_type {
            "content_block_delta" => {
                if let Some((delta_type, content)) = extract_delta(line) {
                    match delta_type {
                        "text_delta" => self.buffer(TextKind::Text, &content, out),
                        "thinking_delta" => self.buffer(TextKind::Thinking, &content, out),
                        "input_json_delta" => self.tool_json.push_str(&content),
                        _ => {}
                    }
                }
            }
            "content_block_start" => {
                let Some(event_str) = extract_event_json(line) else {
                    return;
                };
                let Ok(evt) = serde_json::from_str::<CbStartEvent>(event_str) else {
                    return;
                };
                match evt.content_block.kind {
                    "text" => self.block = Some(TextKind::Text),
                    "thinking" => self.block = Some(TextKind::Thinking),
                    "tool_use" | "server_tool_use" => {
                        let name = evt
                            .content_block
                            .name
                            .map(Cow::into_owned)
                            .unwrap_or_default();
                        self.tool_name = Some(name);
                        self.tool_json.clear();
                        if self.tool_io {
                            let id = evt
                                .content_block
                                .id
                                .map(Cow::into_owned)
                                .unwrap_or_default();
                            self.tool_id = (!id.is_empty()).then_some(id);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => self.end_block(out),
            "message_start" => {
                let Some(event_str) = extract_event_json(line) else {
                    return;
                };
                if let Ok(evt) = serde_json::from_str::<MsgStartEvent>(event_str) {
                    self.input = evt.message.usage.input_tokens;
                    self.cache_read = evt.message.usage.cache_read_input_tokens;
                    self.cache_write = evt.message.usage.cache_creation_input_tokens;
                }
            }
            "message_delta" => {
                let Some(event_str) = extract_event_json(line) else {
                    return;
                };
                if let Ok(evt) = serde_json::from_str::<MsgDeltaEvent>(event_str)
                    && let Some(usage) = evt.usage
                    && usage.output_tokens > 0
                {
                    self.output = usage.output_tokens;
                    self.maybe_emit_tokens(out);
                }
            }
            "error" => {
                let Some(event_str) = extract_event_json(line) else {
                    return;
                };
                if let Ok(evt) = serde_json::from_str::<ErrorEvt>(event_str) {
                    out.push(AgentEvent::Error {
                        error_type: evt.error.error_type.into_owned(),
                        message: evt.error.message.into_owned(),
                    });
                }
            }
            _ => {}
        }
    }

    /// Flush any still-open block at end of stream: a buffered text/thinking tail, or a dangling
    /// `tool_use` whose `content_block_stop` never arrived. Usually empty; exists so a consumer
    /// draining a truncated stream loses nothing.
    pub fn flush(&mut self) -> Vec<AgentEvent> {
        let mut out = Vec::new();
        self.end_block(&mut out);
        out
    }

    /// Append a text/thinking chunk to the open block, emitting one event per completed line;
    /// the sub-line tail stays buffered until the next chunk or the block's stop.
    fn buffer(&mut self, kind: TextKind, chunk: &str, out: &mut Vec<AgentEvent>) {
        self.block = Some(kind);
        self.buf.push_str(chunk);
        let mut start = 0;
        while let Some(pos) = self.buf[start..].find('\n') {
            let end = start + pos;
            out.push(kind.event(self.buf[start..end].to_string()));
            start = end + 1;
        }
        if start > 0 {
            self.buf.drain(..start);
        }
    }

    /// Close the open block: emit a completed tool call, or flush a text/thinking tail.
    fn end_block(&mut self, out: &mut Vec<AgentEvent>) {
        if let Some(name) = self.tool_name.take() {
            let parsed = serde_json::from_str::<Value>(&self.tool_json).ok();
            let summary = parsed
                .as_ref()
                .map(|p| format_tool(&name, p))
                .unwrap_or_default();
            let subagent = name == "Agent" || name == "Task";
            let input = if self.tool_io {
                if let Some(id) = self.tool_id.take() {
                    self.tool_names.insert(id, name.clone());
                }
                parsed.as_ref().map(bounded_input)
            } else {
                None
            };
            out.push(AgentEvent::Tool {
                name,
                summary,
                subagent,
                input,
                result: None,
            });
            self.tool_json.clear();
            return;
        }
        if let Some(kind) = self.block.take()
            && !self.buf.is_empty()
        {
            out.push(kind.event(std::mem::take(&mut self.buf)));
        }
    }

    /// Under verbose tool IO, turn each `tool_result` block claude echoes back in a
    /// `user` message into a Tool event carrying a bounded result excerpt, labeled
    /// with the originating tool's name via the id map. A no-op by default.
    fn tool_results(&mut self, msg: &Value, out: &mut Vec<AgentEvent>) {
        if !self.tool_io {
            return;
        }
        let Some(content) = msg
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(Value::as_array)
        else {
            return;
        };
        for block in content {
            if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let id = str_field(block, "tool_use_id");
            let name = self
                .tool_names
                .remove(&id)
                .unwrap_or_else(|| "tool".to_string());
            out.push(AgentEvent::Tool {
                name,
                summary: "result".to_string(),
                subagent: false,
                input: None,
                result: Some(truncate_chars(&result_text(block), TOOL_IO_LIMIT)),
            });
        }
    }

    /// The collector's running cost, when it beats what the stream has declared for itself. The
    /// agent exports its cost metric every 10 s while `total_cost_usd` lands only at turn end, so
    /// mid-turn this is the only honest number; once the stream reports its own (larger, final)
    /// figure, that one wins.
    fn live_cost(&self) -> Option<f64> {
        self.meters
            .as_ref()
            .map(|m| &m.cost)
            .and_then(CostHandle::get)
            .filter(|live| *live > self.stream_cost)
    }

    /// Emit a [`Tokens`] sample when the running total first appears or grows by a step. `rate`
    /// carries the collector's live rate when one is attached; `None` when telemetry is off.
    fn maybe_emit_tokens(&mut self, out: &mut Vec<AgentEvent>) {
        let total = self.input + self.output + self.cache_read + self.cache_write;
        if self.last_emitted_total != 0
            && total.saturating_sub(self.last_emitted_total) < TOKEN_EMIT_STEP
        {
            return;
        }
        self.last_emitted_total = total;
        out.push(AgentEvent::Tokens(Tokens {
            input: self.input,
            output: self.output,
            cache_read: self.cache_read,
            cache_write: self.cache_write,
            total,
            rate: self
                .meters
                .as_ref()
                .map(|m| &m.rate)
                .and_then(RateHandle::get),
            cost_usd: self.live_cost(),
        }));
    }
}

/// Extract the inner event type from a stream_event line without finding event boundaries.
/// Looks for `"event":{"type":"` which is at a fixed position in the stream_event format.
fn inner_event_type(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    // Fixed layout: {"type":"stream_event","event":{"type":"...
    //                                        byte 31^       ^byte 40
    if bytes.len() > 41 && &bytes[31..40] == b"{\"type\":\"" {
        let val_end = 40 + bytes[40..].iter().position(|&b| b == b'"')?;
        return Some(&line[40..val_end]);
    }
    // Fallback: scan for the pattern
    const NEEDLE: &[u8] = b"\"event\":{\"type\":\"";
    let search_end = bytes.len().min(50);
    let pos = bytes[..search_end]
        .windows(NEEDLE.len())
        .position(|w| w == NEEDLE)?;
    let val_start = pos + NEEDLE.len();
    let val_end = val_start + bytes[val_start..].iter().position(|&b| b == b'"')?;
    Some(&line[val_start..val_end])
}

/// Extract the delta type and content string from a content_block_delta event JSON.
/// Scans for the delta type via bytes, then decodes just the content JSON string,
/// avoiding a full struct parse on the hottest path.
fn extract_delta(event: &str) -> Option<(&str, Cow<'_, str>)> {
    let bytes = event.as_bytes();
    const DELTA_TYPE: &[u8] = b"\"delta\":{\"type\":\"";
    // Start past "content_block_delta" (~byte 60) to skip known prefix
    let skip = 60.min(bytes.len());
    let pos = skip
        + bytes[skip..]
            .windows(DELTA_TYPE.len())
            .position(|w| w == DELTA_TYPE)?;
    let type_start = pos + DELTA_TYPE.len();
    let type_end = type_start + bytes[type_start..].iter().position(|&b| b == b'"')?;
    let delta_type = &event[type_start..type_end];

    let content_key: &[u8] = match delta_type {
        "text_delta" => b"\"text\":",
        "thinking_delta" => b"\"thinking\":",
        "input_json_delta" => b"\"partial_json\":",
        _ => return None,
    };

    let search_start = type_end + 1;
    let key_pos = bytes[search_start..]
        .windows(content_key.len())
        .position(|w| w == content_key)?;
    let value_start = search_start + key_pos + content_key.len();

    if value_start >= bytes.len() || bytes[value_start] != b'"' {
        return None;
    }
    let str_start = value_start + 1;
    let mut i = str_start;
    let mut has_escape = false;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                if !has_escape {
                    return Some((delta_type, Cow::Borrowed(&event[str_start..i])));
                }
                let mut de = serde_json::Deserializer::from_str(&event[value_start..]);
                let value: Cow<'_, str> = serde::Deserialize::deserialize(&mut de).ok()?;
                return Some((delta_type, value));
            }
            b'\\' => {
                has_escape = true;
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Extract the JSON object value of the `"event"` key from a stream_event line.
/// Finds `"event":` then tracks brace nesting (respecting JSON strings) to locate
/// the matching `}`. Returns the substring `{...}` without parsing anything.
fn extract_event_json(line: &str) -> Option<&str> {
    let bytes = line.as_bytes();
    const NEEDLE: &[u8] = b"\"event\":";
    let search_end = bytes.len().min(40);
    let pos = bytes[..search_end]
        .windows(NEEDLE.len())
        .position(|w| w == NEEDLE)?;
    let mut i = pos + NEEDLE.len();
    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != b'{' {
        return None;
    }
    let obj_start = i;
    let mut depth = 0u32;
    let mut in_string = false;
    let mut escape = false;
    while i < bytes.len() {
        let b = bytes[i];
        if escape {
            escape = false;
        } else if in_string {
            if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&line[obj_start..=i]);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Extract the value of the first `"type":"..."` field via byte scanning.
/// Type values are plain ASCII identifiers, never containing escape sequences.
fn quick_type(line: &str) -> Option<&str> {
    const NEEDLE: &[u8] = b"\"type\":\"";
    let bytes = line.as_bytes();
    let search_end = bytes.len().min(80);
    let pos = bytes[..search_end]
        .windows(NEEDLE.len())
        .position(|w| w == NEEDLE)?;
    let val_start = pos + NEEDLE.len();
    let val_end = val_start + bytes[val_start..].iter().position(|&b| b == b'"')?;
    Some(&line[val_start..val_end])
}

/// A compact one-line summary for known tools; unknown tools fall back to a truncated
/// `key=value` join. Cosmetic only.
fn format_tool(name: &str, p: &Value) -> String {
    let get = |keys: &[&str]| -> String {
        for k in keys {
            let v = str_field(p, k);
            if !v.is_empty() {
                return v;
            }
        }
        String::new()
    };
    match name {
        "Bash" => {
            let cmd = get(&["command"]);
            let desc = get(&["description"]);
            if desc.is_empty() {
                format!("$ {cmd}")
            } else {
                format!("$ {cmd}  # {desc}")
            }
        }
        "Read" => {
            let mut s = get(&["file_path", "filePath"]);
            if let Some(o) = p.get("offset") {
                s.push_str(&format!(" L{o}"));
            }
            if let Some(l) = p.get("limit") {
                s.push_str(&format!(" +{l}"));
            }
            s
        }
        "Write" => get(&["file_path", "filePath"]),
        "Edit" => {
            let path = get(&["file_path", "filePath"]);
            let old = get(&["old_string", "oldString"]);
            let first = old.lines().next().unwrap_or("");
            let preview: String = first.chars().take(60).collect();
            let ell = if first.chars().count() > 60 {
                "…"
            } else {
                ""
            };
            format!("{path}: {preview}{ell}")
        }
        "Glob" => {
            let path = get(&["path"]);
            let path = if path.is_empty() { ".".into() } else { path };
            format!("{} in {path}", get(&["pattern"]))
        }
        "Grep" => {
            let path = get(&["path"]);
            let path = if path.is_empty() { ".".into() } else { path };
            format!("/{}/ in {path}", get(&["pattern"]))
        }
        "Agent" | "Task" => {
            let desc = get(&["description"]);
            let at = get(&["subagent_type", "subagentType"]);
            if at.is_empty() {
                desc
            } else {
                format!("[{at}] {desc}")
            }
        }
        "Skill" => format!("/{} {}", get(&["skill"]), get(&["args"]))
            .trim()
            .to_string(),
        _ => {
            let mut parts = Vec::new();
            if let Some(obj) = p.as_object() {
                for (k, v) in obj {
                    let mut val = match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    if val.chars().count() > 60 {
                        val = val.chars().take(60).collect::<String>() + "…";
                    }
                    parts.push(format!("{k}={val}"));
                }
            }
            let joined = parts.join(", ");
            if joined.chars().count() > 200 {
                joined.chars().take(200).collect::<String>() + "…"
            } else {
                joined
            }
        }
    }
}

/// The underlying error of an `api_retry` line. The CLI often reports the literal
/// string "unknown" in `error` while the useful detail sits elsewhere (an error
/// object, or sibling fields), so dig before settling — and when nothing classifies,
/// carry the raw line so the log never says just "unknown".
fn retry_error(msg: &Value) -> String {
    let direct = match msg.get("error") {
        Some(Value::String(s)) => s.clone(),
        Some(obj @ Value::Object(_)) => {
            let m = str_field(obj, "message");
            if m.is_empty() {
                str_field(obj, "type")
            } else {
                m
            }
        }
        _ => String::new(),
    };
    if !direct.is_empty() && direct != "unknown" {
        return truncate_chars(&direct, 300);
    }
    for key in ["message", "status", "reason"] {
        let v = str_field(msg, key);
        if !v.is_empty() {
            return truncate_chars(&v, 300);
        }
    }
    truncate_chars(&msg.to_string(), 300)
}

/// Char-truncate with an ellipsis marker; identity when already within `limit`.
pub(crate) fn truncate_chars(s: &str, limit: usize) -> String {
    if s.chars().count() <= limit {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(limit).collect();
        t.push('…');
        t
    }
}

/// Bound a verbose tool input: pass small inputs through verbatim; large ones become
/// a truncated string of their serialization (truncated JSON is not valid JSON, so it
/// cannot stay a structured value).
pub(crate) fn bounded_input(v: &Value) -> Value {
    let s = v.to_string();
    if s.len() <= TOOL_IO_LIMIT {
        v.clone()
    } else {
        Value::String(truncate_chars(&s, TOOL_IO_LIMIT))
    }
}

/// A `tool_result` block's text: `content` is either a plain string or an array of
/// blocks whose text-typed entries we join.
fn result_text(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(items)) => items
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .map(|b| str_field(b, "text"))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

pub(crate) fn str_field(v: &Value, key: &str) -> String {
    v.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

pub(crate) fn u64_field(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn bool_field(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn f64_field(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn array_len(v: &Value, key: &str) -> u32 {
    v.get(key)
        .and_then(Value::as_array)
        .map_or(0, |a| a.len() as u32)
}

/// An undecoded stdout line, kept verbatim so nothing the agent printed is lost.
fn raw(line: &str) -> AgentEvent {
    AgentEvent::Raw {
        text: line.to_string(),
        stream: RawStream::Stdout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a sequence of lines and collect every event produced, in order.
    fn run(lines: &[&str]) -> Vec<AgentEvent> {
        let mut p = StreamJsonParser::default();
        lines.iter().flat_map(|l| p.push(l)).collect()
    }

    /// The outage shape: claude prints a bare-text failure to stdout and exits non-zero. The
    /// line must reach the caller, not vanish for failing to parse as JSON.
    #[test]
    fn non_json_stdout_passes_through_as_raw() {
        let evs = run(&["Error: connection refused (os error 111)"]);
        assert!(
            matches!(
                evs.as_slice(),
                [AgentEvent::Raw { text, stream: RawStream::Stdout }]
                    if text == "Error: connection refused (os error 111)"
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn blank_stdout_lines_stay_silent() {
        assert!(run(&["", "   "]).is_empty());
    }

    #[test]
    fn unmodeled_json_type_passes_through_as_raw() {
        let line = r#"{"type":"some_future_kind","detail":"x"}"#;
        let evs = run(&[line]);
        assert!(
            matches!(
                evs.as_slice(),
                [AgentEvent::Raw { text, stream: RawStream::Stdout }] if text == line
            ),
            "{evs:?}"
        );
    }

    #[test]
    fn init_reports_model_and_counts() {
        let ev = run(&[
            r#"{"type":"system","subtype":"init","model":"claude-opus-4-8","tools":["a","b","c"],"agents":["x","y"]}"#,
        ]);
        assert!(
            matches!(&ev[..], [AgentEvent::Init { model, tools, agents }]
            if model == "claude-opus-4-8" && *tools == 3 && *agents == 2)
        );
    }

    #[test]
    fn known_ignored_types_emit_nothing() {
        // Real capture shapes the parser deliberately ignores.
        let ev = run(&[
            r#"{"type":"system","subtype":"status","model":"","tools":[]}"#,
            r#"{"type":"rate_limit_event","rate_limit_info":{},"uuid":"u","session_id":"s"}"#,
            r#"{"type":"assistant","message":{"content":[{"type":"text"}]}}"#,
        ]);
        assert!(ev.is_empty(), "no events for ignored types, got {ev:?}");
    }

    #[test]
    fn text_block_buffers_to_lines_and_flushes_tail() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"text","text":""}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"line one\nline "}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"two"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#,
        ]);
        let deltas: Vec<&str> = ev
            .iter()
            .map(|e| match e {
                AgentEvent::Text { delta } => delta.as_str(),
                other => panic!("expected Text, got {other:?}"),
            })
            .collect();
        assert_eq!(deltas, vec!["line one", "line two"]);
    }

    #[test]
    fn thinking_block_maps_to_thinking() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"thinking"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"hmm"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#,
        ]);
        assert!(matches!(&ev[..], [AgentEvent::Thinking { delta }] if delta == "hmm"));
    }

    #[test]
    fn tool_use_accumulates_json_and_emits_on_stop() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":"Edit"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"file_path\":\"p.go\",\"old_string\""}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":":\"snap := x\\nmore\"}"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#,
        ]);
        match &ev[..] {
            [
                AgentEvent::Tool {
                    name,
                    summary,
                    subagent,
                    input,
                    result,
                },
            ] => {
                assert_eq!(name, "Edit");
                assert_eq!(summary, "p.go: snap := x");
                assert!(!subagent);
                assert!(input.is_none(), "compact by default: no input recorded");
                assert!(result.is_none());
            }
            other => panic!("expected one Tool, got {other:?}"),
        }
    }

    #[test]
    fn subagent_tool_is_flagged() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":"Task"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"subagent_type\":\"Explore\",\"description\":\"map it\"}"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#,
        ]);
        assert!(
            matches!(&ev[..], [AgentEvent::Tool { subagent: true, summary, .. }]
            if summary == "[Explore] map it")
        );
    }

    #[test]
    fn tokens_accrue_from_message_start_and_delta() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":3968,"cache_creation_input_tokens":6439,"cache_read_input_tokens":14376,"output_tokens":1}}}}"#,
            r#"{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":4}}}"#,
        ]);
        match &ev[..] {
            [AgentEvent::Tokens(t)] => {
                assert_eq!(t.input, 3968);
                assert_eq!(t.output, 4);
                assert_eq!(t.cache_read, 14376);
                assert_eq!(t.cache_write, 6439);
                assert_eq!(t.total, 3968 + 4 + 14376 + 6439);
                assert!(t.rate.is_none() && t.cost_usd.is_none());
            }
            other => panic!("expected one Tokens, got {other:?}"),
        }
    }

    #[test]
    fn tokens_carry_the_live_rate_when_a_handle_is_attached() {
        let meters = LiveMeters::default();
        let mut p = StreamJsonParser::with_meters(meters.clone());
        let drive = |p: &mut StreamJsonParser| -> Vec<AgentEvent> {
            let mut evs = p.push(
                r#"{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":6000,"output_tokens":1}}}}"#,
            );
            evs.extend(p.push(
                r#"{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":4}}}"#,
            ));
            evs
        };

        // No rate established yet -> None.
        match drive(&mut p).into_iter().find_map(|e| match e {
            AgentEvent::Tokens(t) => Some(t),
            _ => None,
        }) {
            Some(t) => assert!(t.rate.is_none(), "no rate stored yet"),
            None => panic!("expected a tokens sample"),
        }

        // Now a rate is live -> it rides the sample.
        meters.rate.store(318.4);
        let mut p2 = StreamJsonParser::with_meters(meters);
        match drive(&mut p2).into_iter().find_map(|e| match e {
            AgentEvent::Tokens(t) => Some(t),
            _ => None,
        }) {
            Some(t) => assert_eq!(t.rate, Some(318.4)),
            None => panic!("expected a tokens sample"),
        }
    }

    #[test]
    fn tokens_carry_the_collectors_live_cost() {
        let meters = LiveMeters::default();
        let drive = |p: &mut StreamJsonParser| -> Option<Tokens> {
            let mut evs = p.push(
                r#"{"type":"stream_event","event":{"type":"message_start","message":{"usage":{"input_tokens":6000,"output_tokens":1}}}}"#,
            );
            evs.extend(p.push(
                r#"{"type":"stream_event","event":{"type":"message_delta","usage":{"output_tokens":4}}}"#,
            ));
            evs.into_iter().find_map(|e| match e {
                AgentEvent::Tokens(t) => Some(t),
                _ => None,
            })
        };

        // No cost export yet -> nothing to stamp, and the budget falls back to the estimate.
        let mut p = StreamJsonParser::with_meters(meters.clone());
        assert_eq!(drive(&mut p).expect("a tokens sample").cost_usd, None);

        // A live cost the stream has not declared yet wins.
        meters.cost.add(23.4567);
        let mut p2 = StreamJsonParser::with_meters(meters.clone());
        assert_eq!(
            drive(&mut p2).expect("a tokens sample").cost_usd,
            Some(23.4567)
        );

        // Once the stream declares a LARGER figure of its own, that one wins.
        let mut p3 = StreamJsonParser::with_meters(meters);
        p3.push(r#"{"type":"result","subtype":"success","num_turns":1,"total_cost_usd":25.0}"#);
        assert_eq!(drive(&mut p3).expect("a tokens sample").cost_usd, None);
    }

    #[test]
    fn result_carries_cost_and_turns() {
        let ev = run(&[
            r#"{"type":"result","subtype":"success","stop_reason":"end_turn","num_turns":1,"total_cost_usd":0.09208}"#,
        ]);
        match &ev[..] {
            [
                AgentEvent::Result {
                    subtype,
                    is_error,
                    turns,
                    cost_usd,
                    error,
                },
            ] => {
                assert_eq!(subtype, "success");
                assert!(!*is_error, "a clean result is not an error");
                assert!(error.is_none(), "no error text on a clean turn");
                assert_eq!(*turns, 1);
                assert!((cost_usd - 0.09208).abs() < 1e-9);
            }
            other => panic!("expected one Result, got {other:?}"),
        }
    }

    #[test]
    fn result_surfaces_is_error_and_text() {
        // Gotcha: the CLI can send subtype "success" with is_error=true and the reason in
        // `result` (e.g. not logged in). Both must survive decode.
        let ev = run(&[
            r#"{"type":"result","subtype":"success","is_error":true,"num_turns":1,"total_cost_usd":0.0,"result":"Not logged in"}"#,
        ]);
        match &ev[..] {
            [
                AgentEvent::Result {
                    is_error, error, ..
                },
            ] => {
                assert!(*is_error, "the CLI flagged the turn as an error");
                assert_eq!(error.as_deref(), Some("Not logged in"));
            }
            other => panic!("expected one Result, got {other:?}"),
        }
    }

    #[test]
    fn result_flushes_an_open_text_tail() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"text"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"trailing"}}}"#,
            r#"{"type":"result","subtype":"success","num_turns":1,"total_cost_usd":0.5}"#,
        ]);
        assert!(matches!(&ev[..],
            [AgentEvent::Text { delta }, AgentEvent::Result { .. }] if delta == "trailing"));
    }

    #[test]
    fn api_retry_maps_to_retry() {
        let ev = run(&[
            r#"{"type":"system","subtype":"api_retry","attempt":2,"max_retries":10,"error":"overloaded_error"}"#,
        ]);
        assert!(
            matches!(&ev[..], [AgentEvent::Retry { attempt, max, error }]
            if *attempt == 2 && *max == 10 && error == "overloaded_error")
        );
    }

    #[test]
    fn api_retry_digs_past_unknown_for_the_real_error() {
        // error:"unknown" with a sibling message: the message wins.
        let ev = run(&[
            r#"{"type":"system","subtype":"api_retry","attempt":1,"max_retries":15,"error":"unknown","message":"upstream connect error"}"#,
        ]);
        assert!(
            matches!(&ev[..], [AgentEvent::Retry { error, .. }] if error == "upstream connect error")
        );

        // error as an object: its message wins.
        let ev = run(&[
            r#"{"type":"system","subtype":"api_retry","attempt":1,"max_retries":15,"error":{"type":"overloaded_error","message":"busy"}}"#,
        ]);
        assert!(matches!(&ev[..], [AgentEvent::Retry { error, .. }] if error == "busy"));

        // Nothing classifiable: the whole raw line rides along instead of "unknown".
        let ev = run(&[
            r#"{"type":"system","subtype":"api_retry","attempt":1,"max_retries":15,"error":"unknown","delay_ms":2000}"#,
        ]);
        match &ev[..] {
            [AgentEvent::Retry { error, .. }] => {
                assert!(error.contains("delay_ms"), "raw payload carried: {error}");
                assert_ne!(error, "unknown");
            }
            other => panic!("expected one Retry, got {other:?}"),
        }
    }

    #[test]
    fn verbose_tool_io_carries_input_and_result_excerpt() {
        let mut p = StreamJsonParser::default().with_tool_io(true);
        let lines = [
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"command\":\"ls\"}"}}}"#,
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#,
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"a.txt\nb.txt"}]}]}}"#,
        ];
        let ev: Vec<AgentEvent> = lines.iter().flat_map(|l| p.push(l)).collect();
        match &ev[..] {
            [
                AgentEvent::Tool {
                    name: call_name,
                    input,
                    result: call_result,
                    ..
                },
                AgentEvent::Tool {
                    name: res_name,
                    result,
                    ..
                },
            ] => {
                assert_eq!(call_name, "Bash");
                assert_eq!(
                    input
                        .as_ref()
                        .and_then(|i| i.get("command"))
                        .and_then(|c| c.as_str()),
                    Some("ls")
                );
                assert!(call_result.is_none());
                assert_eq!(res_name, "Bash", "result labeled via the id map");
                assert_eq!(result.as_deref(), Some("a.txt\nb.txt"));
            }
            other => panic!("expected call + result Tool events, got {other:?}"),
        }
    }

    #[test]
    fn verbose_tool_io_bounds_oversized_input_and_result() {
        let big = "x".repeat(TOOL_IO_LIMIT * 2);
        let mut p = StreamJsonParser::default().with_tool_io(true);
        let lines = [
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","id":"toolu_2","name":"Write"}}}"#.to_string(),
            format!(
                r#"{{"type":"stream_event","event":{{"type":"content_block_delta","delta":{{"type":"input_json_delta","partial_json":"{{\"content\":\"{big}\"}}"}}}}}}"#
            ),
            r#"{"type":"stream_event","event":{"type":"content_block_stop"}}"#.to_string(),
            format!(
                r#"{{"type":"user","message":{{"content":[{{"type":"tool_result","tool_use_id":"toolu_2","content":"{big}"}}]}}}}"#
            ),
        ];
        let ev: Vec<AgentEvent> = lines.iter().flat_map(|l| p.push(l)).collect();
        match &ev[..] {
            [
                AgentEvent::Tool { input, .. },
                AgentEvent::Tool { result, .. },
            ] => {
                // An oversized input degrades to a truncated string of its serialization.
                let stored = input
                    .as_ref()
                    .and_then(|i| i.as_str())
                    .expect("truncated to string");
                assert!(stored.chars().count() <= TOOL_IO_LIMIT + 1);
                assert!(stored.ends_with('…'));
                let excerpt = result.as_deref().expect("result excerpt");
                assert!(excerpt.chars().count() <= TOOL_IO_LIMIT + 1);
                assert!(excerpt.ends_with('…'));
            }
            other => panic!("expected call + result Tool events, got {other:?}"),
        }
    }

    #[test]
    fn default_mode_ignores_tool_results() {
        let ev = run(&[
            r#"{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"secret-ish output"}]}}"#,
        ]);
        assert!(ev.is_empty(), "compact mode drops tool results, got {ev:?}");
    }

    /// Golden test against a real `claude --output-format stream-json` capture. The parser is
    /// model-agnostic, so this asserts on tool count, text, and cost rather than the model string.
    #[test]
    fn golden_real_hello_capture() {
        let fixture = include_str!("testdata/claude_stream_hello.jsonl");
        let ev = run(&fixture.lines().collect::<Vec<_>>());
        match &ev[..] {
            [
                AgentEvent::Init { tools, .. },
                AgentEvent::Text { delta },
                AgentEvent::Tokens(t),
                AgentEvent::Result {
                    subtype,
                    turns,
                    cost_usd,
                    ..
                },
            ] => {
                assert_eq!(*tools, 29, "real init advertised 29 tools");
                assert_eq!(
                    delta, "hello",
                    "the assistant text, line-flushed at block stop"
                );
                assert_eq!(t.input, 3968);
                assert_eq!(t.cache_read, 14376);
                assert_eq!(t.cache_write, 6439);
                assert_eq!(t.output, 4);
                assert_eq!(t.total, 3968 + 4 + 14376 + 6439);
                assert_eq!(subtype, "success");
                assert_eq!(*turns, 1);
                assert!((cost_usd - 0.09208).abs() < 1e-9, "real total_cost_usd");
            }
            other => panic!("unexpected event sequence from real capture: {other:?}"),
        }
    }

    #[test]
    fn stream_error_maps_to_error() {
        let ev = run(&[
            r#"{"type":"stream_event","event":{"type":"error","error":{"type":"overloaded_error","message":"busy"}}}"#,
        ]);
        assert!(
            matches!(&ev[..], [AgentEvent::Error { error_type, message }]
            if error_type == "overloaded_error" && message == "busy")
        );
    }

    #[test]
    fn flush_drains_an_open_text_tail() {
        let mut p = StreamJsonParser::default();
        let mut ev: Vec<AgentEvent> = Vec::new();
        ev.extend(p.push(
            r#"{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"text"}}}"#,
        ));
        ev.extend(p.push(
            r#"{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"dangling"}}}"#,
        ));
        assert!(ev.is_empty(), "no completed line yet, got {ev:?}");
        let tail = p.flush();
        assert!(matches!(&tail[..], [AgentEvent::Text { delta }] if delta == "dangling"));
        // Idempotent: nothing left to flush.
        assert!(p.flush().is_empty());
    }
}
