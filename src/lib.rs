use hf_hub::api::sync::Api;
use once_cell::sync::Lazy;
use ort::session::Session;
use ort::value::Tensor;
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;
use rusqlite::types::Value;
use rusqlite::OptionalExtension;
use rusqlite::{params, params_from_iter, Connection};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokenizers::{PaddingDirection, PaddingStrategy, TruncationDirection, TruncationStrategy};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use uuid::Uuid;

create_exception!(agent_bus_core, DBBusyError, PyRuntimeError);
create_exception!(agent_bus_core, SchemaMismatchError, PyRuntimeError);
create_exception!(agent_bus_core, TopicNotFoundError, PyRuntimeError);
create_exception!(agent_bus_core, TopicClosedError, PyRuntimeError);
create_exception!(agent_bus_core, TopicMismatchError, PyRuntimeError);

const SCHEMA_VERSION: &str = "6";
const DEFAULT_EMBEDDING_MODEL: &str = "BAAI/bge-small-en-v1.5";
const DEFAULT_MAX_TOKENS: usize = 512;
const MAX_EMBEDDING_MAX_TOKENS: usize = 8192;
const DEFAULT_ONNX_FILE: &str = "onnx/model.onnx";
const FALLBACK_ONNX_FILE: &str = "model.onnx";
const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
#[no_mangle]
pub extern "C" fn __aarch64_cas8_sync(expected: u64, desired: u64, ptr: *mut u64) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};

    let atomic_ptr = ptr.cast::<AtomicU64>();
    // SAFETY: This function is a compatibility shim for prebuilt dependencies that expect the
    // `__aarch64_cas8_sync` symbol (notably ONNX Runtime / XNNPACK on Linux aarch64). The pointer
    // is expected to be valid and properly aligned for 64-bit atomic operations.
    debug_assert_eq!(
        (ptr as usize) % std::mem::align_of::<AtomicU64>(),
        0,
        "unaligned __aarch64_cas8_sync pointer"
    );
    unsafe {
        match (*atomic_ptr).compare_exchange(expected, desired, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(prev) => prev,
            Err(actual) => actual,
        }
    }
}

struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

type EmbedderCell = Arc<OnceLock<Arc<Mutex<Embedder>>>>;

static EMBEDDERS: Lazy<Mutex<HashMap<String, EmbedderCell>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
struct TopicRow {
    topic_id: String,
    name: String,
    status: String,
    created_at: f64,
    closed_at: Option<f64>,
    close_reason: Option<String>,
    metadata_json: Option<String>,
}

#[derive(Debug)]
struct MessageRow {
    message_id: String,
    topic_id: String,
    seq: i64,
    sender: String,
    message_type: String,
    reply_to: Option<String>,
    content_markdown: String,
    metadata_json: Option<String>,
    client_message_id: Option<String>,
    created_at: f64,
}

#[derive(Debug)]
struct FtsRow {
    topic_id: String,
    topic_name: String,
    message_id: String,
    seq: i64,
    sender: String,
    message_type: String,
    created_at: f64,
    snippet: String,
    rank: f64,
    content_markdown: Option<String>,
}

#[derive(Debug)]
struct EmbeddingMessageRow {
    message_id: String,
    topic_id: String,
    content_markdown: String,
    created_at: f64,
}

#[derive(Debug)]
struct ChunkCandidateRow {
    message_id: String,
    topic_id: String,
    topic_name: String,
    seq: i64,
    sender: String,
    message_type: String,
    created_at: f64,
    chunk_index: i64,
    start_char: i64,
    end_char: i64,
    dims: i64,
    vector: Vec<u8>,
}

#[derive(Debug)]
struct TopicCountRow {
    topic_id: String,
    name: String,
    status: String,
    created_at: f64,
    closed_at: Option<f64>,
    close_reason: Option<String>,
    metadata_json: Option<String>,
    message_count: i64,
    last_seq: i64,
}

#[derive(Debug)]
struct CursorRow {
    topic_id: String,
    agent_name: String,
    last_seq: i64,
    updated_at: f64,
}

struct OutboxItem {
    content_markdown: String,
    message_type: String,
    reply_to: Option<String>,
    metadata_json: Option<String>,
    client_message_id: Option<String>,
}

struct EmbeddingChunk {
    chunk_index: i64,
    start_char: i64,
    end_char: i64,
    vector: Vec<u8>,
    text_hash: String,
}

impl<'a, 'py> FromPyObject<'a, 'py> for OutboxItem {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let dict = obj.cast::<PyDict>().map_err(PyErr::from)?.to_owned();
        let content_markdown = dict
            .get_item("content_markdown")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("content_markdown"))?
            .extract()?;
        let message_type = dict
            .get_item("message_type")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("message_type"))?
            .extract()?;
        let reply_to = dict.get_item("reply_to")?.and_then(|v| v.extract().ok());
        let metadata_json = dict
            .get_item("metadata_json")?
            .and_then(|v| v.extract().ok());
        let client_message_id = dict
            .get_item("client_message_id")?
            .and_then(|v| v.extract().ok());
        Ok(Self {
            content_markdown,
            message_type,
            reply_to,
            metadata_json,
            client_message_id,
        })
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for EmbeddingChunk {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let dict = obj.cast::<PyDict>().map_err(PyErr::from)?.to_owned();
        Ok(Self {
            chunk_index: dict
                .get_item("chunk_index")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("chunk_index"))?
                .extract()?,
            start_char: dict
                .get_item("start_char")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("start_char"))?
                .extract()?,
            end_char: dict
                .get_item("end_char")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("end_char"))?
                .extract()?,
            vector: dict
                .get_item("vector")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("vector"))?
                .extract()?,
            text_hash: dict
                .get_item("text_hash")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("text_hash"))?
                .extract()?,
        })
    }
}

impl Embedder {
    fn load(model_id: &str, max_tokens: usize) -> PyResult<Self> {
        let (model_path, tokenizer_path) = resolve_embedding_paths(model_id)?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to load tokenizer from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        if max_tokens == 0 {
            return Err(PyValueError::new_err("max_tokens must be > 0"));
        }

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| {
                PyRuntimeError::new_err(format!("failed to set tokenizer truncation: {e}"))
            })?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        let session = Session::builder()
            .map_err(map_ort_error)?
            .commit_from_file(&model_path)
            .map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "failed to load ONNX model from {}: {e}",
                    model_path.display()
                ))
            })?;

        Ok(Self { session, tokenizer })
    }

    fn embed(&mut self, texts: &[String], normalize: bool) -> PyResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let inputs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| PyRuntimeError::new_err(format!("tokenization failed: {e}")))?;

        if encodings.is_empty() {
            return Ok(Vec::new());
        }

        let seq_len = encodings[0].get_ids().len();
        if seq_len == 0 {
            return Ok(vec![Vec::new(); encodings.len()]);
        }

        let batch = encodings.len();
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut token_type_ids: Vec<i64> = Vec::with_capacity(batch * seq_len);

        for enc in &encodings {
            if enc.get_ids().len() != seq_len {
                return Err(PyRuntimeError::new_err(
                    "tokenizer returned varying sequence lengths; enable padding",
                ));
            }
            input_ids.extend(enc.get_ids().iter().map(|v| *v as i64));
            attention_mask.extend(enc.get_attention_mask().iter().map(|v| *v as i64));
            token_type_ids.extend(enc.get_type_ids().iter().map(|v| *v as i64));
        }

        let input_ids_tensor: Tensor<i64> =
            Tensor::from_array(([batch, seq_len], input_ids)).map_err(map_ort_error)?;
        let attention_tensor: Tensor<i64> =
            Tensor::from_array(([batch, seq_len], attention_mask)).map_err(map_ort_error)?;
        let token_type_tensor: Tensor<i64> =
            Tensor::from_array(([batch, seq_len], token_type_ids)).map_err(map_ort_error)?;

        let mut input_ids_opt = Some(input_ids_tensor);
        let mut attention_opt = Some(attention_tensor);
        let mut token_type_opt = Some(token_type_tensor);

        let mut inputs = Vec::new();
        for input in self.session.inputs() {
            match input.name() {
                "input_ids" => {
                    inputs.push((
                        "input_ids".to_string(),
                        input_ids_opt
                            .take()
                            .ok_or_else(|| PyRuntimeError::new_err("input_ids already consumed"))?,
                    ));
                }
                "attention_mask" => {
                    inputs.push((
                        "attention_mask".to_string(),
                        attention_opt.take().ok_or_else(|| {
                            PyRuntimeError::new_err("attention_mask already consumed")
                        })?,
                    ));
                }
                "token_type_ids" => {
                    inputs.push((
                        "token_type_ids".to_string(),
                        token_type_opt.take().ok_or_else(|| {
                            PyRuntimeError::new_err("token_type_ids already consumed")
                        })?,
                    ));
                }
                other => {
                    return Err(PyRuntimeError::new_err(format!(
                        "unsupported model input: {other}"
                    )));
                }
            }
        }

        if input_ids_opt.is_some() {
            return Err(PyRuntimeError::new_err(
                "model inputs did not include required input_ids",
            ));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| PyRuntimeError::new_err(format!("onnx inference failed: {e}")))?;

        let output = if outputs.contains_key("sentence_embedding") {
            outputs.get("sentence_embedding").unwrap()
        } else if outputs.contains_key("embeddings") {
            outputs.get("embeddings").unwrap()
        } else if outputs.contains_key("pooler_output") {
            outputs.get("pooler_output").unwrap()
        } else if outputs.contains_key("last_hidden_state") {
            outputs.get("last_hidden_state").unwrap()
        } else {
            &outputs[0]
        };

        let (shape, data) = output.try_extract_tensor::<f32>().map_err(map_ort_error)?;
        let dims: &[i64] = shape;

        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch);
        match dims.len() {
            2 => {
                let hidden = dims[1] as usize;
                if data.len() < batch * hidden {
                    return Err(PyRuntimeError::new_err("unexpected output shape"));
                }
                for i in 0..batch {
                    let start = i * hidden;
                    let end = start + hidden;
                    let mut vec = data[start..end].to_vec();
                    if normalize {
                        l2_normalize(&mut vec);
                    }
                    embeddings.push(vec);
                }
            }
            3 => {
                let seq = dims[1] as usize;
                let hidden = dims[2] as usize;
                let stride = seq * hidden;
                if data.len() < batch * stride {
                    return Err(PyRuntimeError::new_err("unexpected output shape"));
                }
                for i in 0..batch {
                    let base = i * stride;
                    let start = base;
                    let end = start + hidden;
                    let mut vec = data[start..end].to_vec();
                    if normalize {
                        l2_normalize(&mut vec);
                    }
                    embeddings.push(vec);
                }
            }
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported output rank: {}",
                    dims.len()
                )));
            }
        }

        Ok(embeddings)
    }
}

fn embedding_cache_key(model_id: &str, max_tokens: usize) -> String {
    format!("{model_id}::max={max_tokens}")
}

fn get_embedder(model_id: &str, max_tokens: usize) -> PyResult<Arc<Mutex<Embedder>>> {
    let key = embedding_cache_key(model_id, max_tokens);
    let cell = {
        let mut cache = EMBEDDERS
            .lock()
            .map_err(|_| PyRuntimeError::new_err("embedding cache lock poisoned"))?;
        cache
            .entry(key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    };

    let embedder = cell.get_or_try_init(|| {
        let embedder = Embedder::load(model_id, max_tokens)?;
        Ok(Arc::new(Mutex::new(embedder)))
    })?;
    Ok(Arc::clone(embedder))
}

fn resolve_embedding_paths(model_id: &str) -> PyResult<(PathBuf, PathBuf)> {
    let onnx_override = std::env::var("AGENT_BUS_EMBEDDING_ONNX_PATH").ok();
    let tokenizer_override = std::env::var("AGENT_BUS_EMBEDDING_TOKENIZER_PATH").ok();

    let tokenizer_path = if let Some(path) = tokenizer_override {
        let raw = expand_home(&path).unwrap_or(path);
        let resolved = PathBuf::from(raw);
        if !resolved.exists() {
            return Err(PyRuntimeError::new_err(format!(
                "tokenizer path not found: {}",
                resolved.display()
            )));
        }
        Some(resolved)
    } else {
        None
    };

    if let Some(path) = onnx_override {
        let raw = expand_home(&path).unwrap_or(path);
        let resolved = PathBuf::from(raw);
        if !resolved.exists() {
            return Err(PyRuntimeError::new_err(format!(
                "ONNX model path not found: {}",
                resolved.display()
            )));
        }

        let tok_path = if let Some(tok) = tokenizer_path {
            tok
        } else {
            let candidate = resolved
                .parent()
                .map(|p| p.join(DEFAULT_TOKENIZER_FILE))
                .filter(|p| p.exists());
            if let Some(p) = candidate {
                p
            } else {
                return Err(PyRuntimeError::new_err(
                    "tokenizer path not found; set AGENT_BUS_EMBEDDING_TOKENIZER_PATH",
                ));
            }
        };

        return Ok((resolved, tok_path));
    }

    let api =
        Api::new().map_err(|e| PyRuntimeError::new_err(format!("failed to init hf-hub: {e}")))?;
    let repo = api.model(model_id.to_string());

    let tokenizer_file = std::env::var("AGENT_BUS_EMBEDDING_TOKENIZER_FILE")
        .ok()
        .unwrap_or_else(|| DEFAULT_TOKENIZER_FILE.to_string());
    let tokenizer_path = repo.get(&tokenizer_file).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to download tokenizer {tokenizer_file} from {model_id}: {e}"
        ))
    })?;

    let mut candidates = Vec::new();
    if let Ok(file) = std::env::var("AGENT_BUS_EMBEDDING_ONNX_FILE") {
        candidates.push(file);
    }
    candidates.push(DEFAULT_ONNX_FILE.to_string());
    candidates.push(FALLBACK_ONNX_FILE.to_string());

    let mut last_err = None;
    for file in candidates {
        match repo.get(&file) {
            Ok(path) => return Ok((path, tokenizer_path.clone())),
            Err(e) => last_err = Some((file, e)),
        }
    }

    if let Some((file, err)) = last_err {
        return Err(PyRuntimeError::new_err(format!(
            "failed to download ONNX model {file} from {model_id}: {err}"
        )));
    }
    Err(PyRuntimeError::new_err("failed to resolve ONNX model"))
}

fn default_max_tokens() -> usize {
    std::env::var("AGENT_BUS_EMBEDDING_MAX_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0 && *v <= MAX_EMBEDDING_MAX_TOKENS)
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

fn l2_normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec {
            *v /= norm;
        }
    }
}

fn map_ort_error(err: ort::Error) -> PyErr {
    PyRuntimeError::new_err(format!("onnx runtime error: {err}"))
}

#[pyfunction]
#[pyo3(signature = (texts, model=None, normalize=True, max_tokens=None))]
fn embed_texts(
    py: Python<'_>,
    texts: Vec<String>,
    model: Option<String>,
    normalize: bool,
    max_tokens: Option<usize>,
) -> PyResult<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let model_id = model.unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string());
    let max_tokens = max_tokens.unwrap_or_else(default_max_tokens);
    let embedder = get_embedder(&model_id, max_tokens)?;
    py.detach(|| {
        let mut guard = embedder
            .lock()
            .map_err(|_| PyRuntimeError::new_err("embedding lock poisoned"))?;
        guard.embed(&texts, normalize)
    })
}

#[pyclass]
struct CoreDb {
    path: String,
    fts_available: Mutex<Option<bool>>,
}

#[pymethods]
impl CoreDb {
    #[new]
    fn new(path: Option<String>) -> PyResult<Self> {
        let resolved = resolve_db_path(path);
        Ok(Self {
            path: resolved,
            fts_available: Mutex::new(None),
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.path.clone()
    }

    #[getter]
    fn fts_available(&self) -> bool {
        self.fts_available
            .lock()
            .ok()
            .and_then(|v| *v)
            .unwrap_or(false)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (query, topic_id=None, sender=None, message_type=None, limit=20, include_content=false))]
    fn search_messages_fts(
        &self,
        py: Python<'_>,
        query: String,
        topic_id: Option<String>,
        sender: Option<String>,
        message_type: Option<String>,
        limit: usize,
        include_content: bool,
    ) -> PyResult<Py<PyAny>> {
        if query.trim().is_empty() {
            return Err(PyValueError::new_err("query must be a non-empty string"));
        }
        if limit == 0 {
            return Err(PyValueError::new_err("limit must be > 0"));
        }

        let conn = self.connect()?;

        let mut stmt = match conn.prepare("SELECT 1 FROM messages_fts LIMIT 1") {
            Ok(stmt) => stmt,
            Err(err) => {
                if is_fts_unavailable(&err) {
                    return Err(PyRuntimeError::new_err(
                        "FTS5 is not available on this SQLite build (cannot search).",
                    ));
                }
                return Err(map_db_error(err));
            }
        };
        match stmt.query_row([], |_| Ok(())) {
            Ok(()) | Err(rusqlite::Error::QueryReturnedNoRows) => {}
            Err(err) => {
                if is_fts_unavailable(&err) {
                    return Err(PyRuntimeError::new_err(
                        "FTS5 is not available on this SQLite build (cannot search).",
                    ));
                }
                return Err(map_db_error(err));
            }
        }

        let mut where_parts = vec!["messages_fts MATCH ?".to_string()];
        let mut params: Vec<Value> = vec![Value::from(query)];

        if let Some(t) = topic_id {
            where_parts.push("m.topic_id = ?".to_string());
            params.push(Value::from(t));
        }
        if let Some(s) = sender {
            where_parts.push("m.sender = ?".to_string());
            params.push(Value::from(s));
        }
        if let Some(mt) = message_type {
            where_parts.push("m.message_type = ?".to_string());
            params.push(Value::from(mt));
        }
        params.push(Value::from(limit as i64));

        let content_col = if include_content {
            ", m.content_markdown AS content_markdown"
        } else {
            ""
        };

        let sql = format!(
            "
            SELECT
              m.topic_id,
              t.name AS topic_name,
              m.message_id,
              m.seq,
              m.sender,
              m.message_type,
              m.created_at,
              snippet(messages_fts, 0, '[', ']', '…', 10) AS snippet,
              bm25(messages_fts) AS rank
              {content_col}
            FROM messages_fts
            JOIN messages m ON m.rowid = messages_fts.rowid
            JOIN topics t ON t.topic_id = m.topic_id
            WHERE {where}
            ORDER BY rank ASC
            LIMIT ?
            ",
            where = where_parts.join(" AND "),
        );

        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(params.iter()), |row| {
                Ok(FtsRow {
                    topic_id: row.get("topic_id")?,
                    topic_name: row.get("topic_name")?,
                    message_id: row.get("message_id")?,
                    seq: row.get("seq")?,
                    sender: row.get("sender")?,
                    message_type: row.get("message_type")?,
                    created_at: row.get("created_at")?,
                    snippet: row.get("snippet")?,
                    rank: row.get("rank")?,
                    content_markdown: if include_content {
                        Some(row.get("content_markdown")?)
                    } else {
                        None
                    },
                })
            })
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            let dict = PyDict::new(py);
            dict.set_item("topic_id", row.topic_id)?;
            dict.set_item("topic_name", row.topic_name)?;
            dict.set_item("message_id", row.message_id)?;
            dict.set_item("seq", row.seq)?;
            dict.set_item("sender", row.sender)?;
            dict.set_item("message_type", row.message_type)?;
            dict.set_item("created_at", row.created_at)?;
            dict.set_item("snippet", row.snippet)?;
            dict.set_item("rank", row.rank)?;
            if let Some(content) = row.content_markdown {
                dict.set_item("content_markdown", content)?;
            }
            out.append(dict)?;
        }
        Ok(out.into())
    }

    fn get_message_by_id(&self, py: Python<'_>, message_id: String) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let row = conn
            .query_row(
                "
                SELECT
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                FROM messages
                WHERE message_id = ?
                ",
                params![message_id],
                message_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        let msg =
            row.ok_or_else(|| PyValueError::new_err(format!("message not found: {message_id}")))?;
        Ok(message_to_dict(py, &msg))
    }

    #[allow(clippy::too_many_arguments)]
    fn upsert_embeddings(
        &self,
        message_id: String,
        model: String,
        topic_id: String,
        content_hash: String,
        chunk_size: i64,
        chunk_overlap: i64,
        dims: i64,
        chunks: Vec<EmbeddingChunk>,
    ) -> PyResult<()> {
        let mut conn = self.connect()?;
        let updated_at = now();
        let tx = conn.transaction().map_err(map_db_error)?;

        tx.execute(
            "DELETE FROM chunk_embeddings WHERE message_id = ? AND model = ?",
            params![message_id, model],
        )
        .map_err(map_db_error)?;

        for c in chunks {
            tx.execute(
                "
                INSERT INTO chunk_embeddings(
                  message_id, model, topic_id, chunk_index, start_char, end_char,
                  dims, vector, text_hash, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ",
                params![
                    message_id,
                    model,
                    topic_id,
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    dims,
                    c.vector,
                    c.text_hash,
                    updated_at
                ],
            )
            .map_err(map_db_error)?;
        }

        tx.execute(
            "
            INSERT INTO message_embedding_state(
              message_id, model, content_hash, chunk_size, chunk_overlap, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id, model) DO UPDATE SET
              content_hash = excluded.content_hash,
              chunk_size = excluded.chunk_size,
              chunk_overlap = excluded.chunk_overlap,
              updated_at = excluded.updated_at
            ",
            params![
                message_id,
                model,
                content_hash,
                chunk_size,
                chunk_overlap,
                updated_at
            ],
        )
        .map_err(map_db_error)?;

        tx.commit().map_err(map_db_error)?;
        Ok(())
    }

    fn get_embedding_state(
        &self,
        py: Python<'_>,
        message_id: String,
        model: String,
    ) -> PyResult<Option<Py<PyAny>>> {
        let conn = self.connect()?;
        let row = conn
            .query_row(
                "
                SELECT message_id, model, content_hash, chunk_size, chunk_overlap, updated_at
                FROM message_embedding_state
                WHERE message_id = ? AND model = ?
                ",
                params![message_id, model],
                |r| {
                    Ok((
                        r.get::<_, String>("message_id")?,
                        r.get::<_, String>("model")?,
                        r.get::<_, String>("content_hash")?,
                        r.get::<_, i64>("chunk_size")?,
                        r.get::<_, i64>("chunk_overlap")?,
                        r.get::<_, f64>("updated_at")?,
                    ))
                },
            )
            .optional()
            .map_err(map_db_error)?;

        let Some((mid, model, content_hash, chunk_size, chunk_overlap, updated_at)) = row else {
            return Ok(None);
        };

        let dict = PyDict::new(py);
        dict.set_item("message_id", mid)?;
        dict.set_item("model", model)?;
        dict.set_item("content_hash", content_hash)?;
        dict.set_item("chunk_size", chunk_size)?;
        dict.set_item("chunk_overlap", chunk_overlap)?;
        dict.set_item("updated_at", updated_at)?;
        Ok(Some(dict.into()))
    }

    #[pyo3(signature = (topic_id=None, limit=1000))]
    fn list_messages_for_embedding(
        &self,
        py: Python<'_>,
        topic_id: Option<String>,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        if limit == 0 {
            return Err(PyValueError::new_err("limit must be > 0"));
        }
        let conn = self.connect()?;
        let mut where_sql = String::new();
        let mut params: Vec<Value> = Vec::new();
        if let Some(t) = topic_id {
            where_sql = "WHERE topic_id = ?".to_string();
            params.push(Value::from(t));
        }
        params.push(Value::from(limit as i64));

        let sql = format!(
            "
            SELECT message_id, topic_id, content_markdown, created_at
            FROM messages
            {where_sql}
            ORDER BY created_at ASC
            LIMIT ?
            "
        );
        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(params.iter()), |row| {
                Ok(EmbeddingMessageRow {
                    message_id: row.get("message_id")?,
                    topic_id: row.get("topic_id")?,
                    content_markdown: row.get("content_markdown")?,
                    created_at: row.get("created_at")?,
                })
            })
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            let dict = PyDict::new(py);
            dict.set_item("message_id", row.message_id)?;
            dict.set_item("topic_id", row.topic_id)?;
            dict.set_item("content_markdown", row.content_markdown)?;
            dict.set_item("created_at", row.created_at)?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    fn enqueue_embedding_jobs(
        &self,
        jobs: Vec<(String, String)>,
        model: String,
    ) -> PyResult<usize> {
        if jobs.is_empty() {
            return Ok(0);
        }
        let mut conn = self.connect()?;
        let updated_at = now();
        let created_at = updated_at;
        let tx = conn.transaction().map_err(map_db_error)?;
        for (message_id, topic_id) in &jobs {
            tx.execute(
                "
                INSERT INTO embedding_jobs(
                  message_id, model, topic_id, status, attempts, locked_by, locked_at,
                  last_error, created_at, updated_at
                )
                VALUES (?, ?, ?, 'pending', 0, NULL, NULL, NULL, ?, ?)
                ON CONFLICT(message_id, model) DO UPDATE SET
                  topic_id = excluded.topic_id,
                  status = 'pending',
                  attempts = 0,
                  locked_by = NULL,
                  locked_at = NULL,
                  last_error = NULL,
                  updated_at = excluded.updated_at
                ",
                params![message_id, &model, topic_id, created_at, updated_at],
            )
            .map_err(map_db_error)?;
        }
        tx.commit().map_err(map_db_error)?;
        Ok(jobs.len())
    }

    #[allow(clippy::too_many_arguments)]
    fn claim_embedding_jobs(
        &self,
        py: Python<'_>,
        model: String,
        limit: usize,
        worker_id: String,
        lock_ttl_seconds: i64,
        error_retry_seconds: i64,
        max_attempts: i64,
    ) -> PyResult<Py<PyAny>> {
        if limit == 0 {
            return Err(PyValueError::new_err("limit must be > 0"));
        }
        if lock_ttl_seconds <= 0 {
            return Err(PyValueError::new_err("lock_ttl_seconds must be > 0"));
        }
        if error_retry_seconds < 0 {
            return Err(PyValueError::new_err("error_retry_seconds must be >= 0"));
        }
        if max_attempts <= 0 {
            return Err(PyValueError::new_err("max_attempts must be > 0"));
        }

        let conn = self.connect()?;
        conn.execute_batch("BEGIN IMMEDIATE")
            .map_err(map_db_error)?;

        let claimed_at = now();
        let stale_before = claimed_at - lock_ttl_seconds as f64;
        let error_ready_before = claimed_at - error_retry_seconds as f64;

        let mut stmt = conn
            .prepare(
                "
                SELECT message_id, topic_id
                FROM embedding_jobs
                WHERE model = ?
                  AND attempts < ?
                  AND (
                    status = 'pending'
                    OR (status = 'error' AND updated_at <= ?)
                    OR (status = 'processing' AND (locked_at IS NULL OR locked_at <= ?))
                  )
                ORDER BY updated_at ASC
                LIMIT ?
                ",
            )
            .map_err(map_db_error)?;

        let rows: Vec<(String, String)> = stmt
            .query_map(
                params![
                    model,
                    max_attempts,
                    error_ready_before,
                    stale_before,
                    limit as i64
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        if rows.is_empty() {
            conn.execute_batch("COMMIT").map_err(map_db_error)?;
            return Ok(PyList::empty(py).into());
        }

        let message_ids: Vec<String> = rows.iter().map(|(m, _)| m.clone()).collect();
        let placeholders = placeholders(message_ids.len());

        let mut params_vec: Vec<Value> = vec![
            Value::from(worker_id.clone()),
            Value::from(claimed_at),
            Value::from(claimed_at),
            Value::from(model.clone()),
            Value::from(max_attempts),
            Value::from(error_ready_before),
            Value::from(stale_before),
        ];
        for mid in &message_ids {
            params_vec.push(Value::from(mid.clone()));
        }

        let update_sql = format!(
            "
            UPDATE embedding_jobs
            SET
              status = 'processing',
              locked_by = ?,
              locked_at = ?,
              updated_at = ?,
              attempts = attempts + 1
            WHERE model = ?
              AND attempts < ?
              AND (
                status = 'pending'
                OR (status = 'error' AND updated_at <= ?)
                OR (status = 'processing' AND (locked_at IS NULL OR locked_at <= ?))
              )
              AND message_id IN ({placeholders})
            "
        );

        conn.execute(&update_sql, params_from_iter(params_vec.iter()))
            .map_err(map_db_error)?;

        let mut stmt = conn
            .prepare(
                "
                SELECT message_id, topic_id, attempts
                FROM embedding_jobs
                WHERE model = ? AND locked_by = ? AND locked_at = ?
                ",
            )
            .map_err(map_db_error)?;
        let claimed_rows = stmt
            .query_map(params![model, worker_id, claimed_at], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        conn.execute_batch("COMMIT").map_err(map_db_error)?;

        let out = PyList::empty(py);
        for (mid, tid, attempts) in claimed_rows {
            let dict = PyDict::new(py);
            dict.set_item("message_id", mid)?;
            dict.set_item("topic_id", tid)?;
            dict.set_item("attempts", attempts)?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    fn complete_embedding_job(&self, message_id: String, model: String) -> PyResult<()> {
        let conn = self.connect()?;
        let updated_at = now();
        conn.execute(
            "
            UPDATE embedding_jobs
            SET status = 'done',
                locked_by = NULL,
                locked_at = NULL,
                last_error = NULL,
                updated_at = ?
            WHERE message_id = ? AND model = ?
            ",
            params![updated_at, message_id, model],
        )
        .map_err(map_db_error)?;
        Ok(())
    }

    fn fail_embedding_job(&self, message_id: String, model: String, error: String) -> PyResult<()> {
        let conn = self.connect()?;
        let updated_at = now();
        conn.execute(
            "
            UPDATE embedding_jobs
            SET status = 'error',
                locked_by = NULL,
                locked_at = NULL,
                last_error = ?,
                updated_at = ?
            WHERE message_id = ? AND model = ?
            ",
            params![error, updated_at, message_id, model],
        )
        .map_err(map_db_error)?;
        Ok(())
    }

    #[pyo3(signature = (model, topic_id=None, message_ids=None))]
    fn list_chunk_embedding_candidates(
        &self,
        py: Python<'_>,
        model: String,
        topic_id: Option<String>,
        message_ids: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let mut where_parts = vec!["e.model = ?".to_string()];
        let mut params: Vec<Value> = vec![Value::from(model)];

        if let Some(tid) = topic_id {
            where_parts.push("e.topic_id = ?".to_string());
            params.push(Value::from(tid));
        }
        if let Some(ids) = message_ids {
            if ids.is_empty() {
                return Ok(PyList::empty(py).into());
            }
            let placeholders = placeholders(ids.len());
            where_parts.push(format!("e.message_id IN ({placeholders})"));
            for mid in ids {
                params.push(Value::from(mid));
            }
        }

        let sql = format!(
            "
            SELECT
              e.message_id,
              e.topic_id,
              t.name AS topic_name,
              m.seq,
              m.sender,
              m.message_type,
              m.created_at,
              e.chunk_index,
              e.start_char,
              e.end_char,
              e.dims,
              e.vector
            FROM chunk_embeddings e
            JOIN messages m ON m.message_id = e.message_id
            JOIN topics t ON t.topic_id = e.topic_id
            WHERE {where_clause}
            ",
            where_clause = where_parts.join(" AND ")
        );

        let conn = self.connect()?;
        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(params.iter()), |row| {
                Ok(ChunkCandidateRow {
                    message_id: row.get("message_id")?,
                    topic_id: row.get("topic_id")?,
                    topic_name: row.get("topic_name")?,
                    seq: row.get("seq")?,
                    sender: row.get("sender")?,
                    message_type: row.get("message_type")?,
                    created_at: row.get("created_at")?,
                    chunk_index: row.get("chunk_index")?,
                    start_char: row.get("start_char")?,
                    end_char: row.get("end_char")?,
                    dims: row.get("dims")?,
                    vector: row.get("vector")?,
                })
            })
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            let dict = PyDict::new(py);
            dict.set_item("message_id", row.message_id)?;
            dict.set_item("topic_id", row.topic_id)?;
            dict.set_item("topic_name", row.topic_name)?;
            dict.set_item("seq", row.seq)?;
            dict.set_item("sender", row.sender)?;
            dict.set_item("message_type", row.message_type)?;
            dict.set_item("created_at", row.created_at)?;
            dict.set_item("chunk_index", row.chunk_index)?;
            dict.set_item("start_char", row.start_char)?;
            dict.set_item("end_char", row.end_char)?;
            dict.set_item("dims", row.dims)?;
            let vector = PyBytes::new(py, &row.vector);
            dict.set_item("vector", vector)?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    fn get_topic(&self, py: Python<'_>, topic_id: String) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let row = conn
            .query_row(
                "
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE topic_id = ?
                ",
                params![topic_id],
                topic_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        let topic = row.ok_or_else(|| TopicNotFoundError::new_err(topic_id))?;
        Ok(topic_to_dict(py, &topic))
    }

    #[pyo3(signature = (name=None, metadata_json=None, mode=None, now_override=None))]
    fn topic_create(
        &self,
        py: Python<'_>,
        name: Option<String>,
        metadata_json: Option<String>,
        mode: Option<String>,
        now_override: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let mode = mode.unwrap_or_else(|| "new".to_string());
        if mode != "reuse" && mode != "new" {
            return Err(PyValueError::new_err("mode must be 'reuse' or 'new'"));
        }
        let topic_id = new_id();
        let topic_name = name.unwrap_or_else(|| format!("topic-{topic_id}"));
        let created_at = now_override.unwrap_or_else(now);

        let mut conn = self.connect()?;
        if mode == "reuse" {
            let row = conn
                .query_row(
                    "
                    SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                    FROM topics
                    WHERE name = ? AND status = 'open'
                    ORDER BY created_at DESC
                    LIMIT 1
                    ",
                    params![topic_name],
                    topic_row_from,
                )
                .optional()
                .map_err(map_db_error)?;
            if let Some(existing) = row {
                return Ok(topic_to_dict(py, &existing));
            }
        }

        let tx = conn.transaction().map_err(map_db_error)?;
        tx.execute(
            "
            INSERT INTO topics(topic_id, name, created_at, status, closed_at, close_reason, metadata_json)
            VALUES (?, ?, ?, 'open', NULL, NULL, ?)
            ",
            params![topic_id, topic_name, created_at, metadata_json],
        )
        .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;

        let topic = TopicRow {
            topic_id,
            name: topic_name,
            status: "open".to_string(),
            created_at,
            closed_at: None,
            close_reason: None,
            metadata_json,
        };
        Ok(topic_to_dict(py, &topic))
    }

    fn topic_list(&self, py: Python<'_>, status: String) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let mut where_sql = String::new();
        let mut params: Vec<Value> = vec![];
        if status == "open" || status == "closed" {
            where_sql = "WHERE status = ?".to_string();
            params.push(Value::from(status));
        }

        let sql = format!(
            "
            SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
            FROM topics
            {where_sql}
            ORDER BY created_at DESC
            "
        );

        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(params.iter()), topic_row_from)
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            out.append(topic_to_dict(py, &row.map_err(map_db_error)?))?;
        }
        Ok(out.into())
    }

    fn topic_list_with_counts(
        &self,
        py: Python<'_>,
        status: String,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let mut where_sql = String::new();
        let mut params: Vec<Value> = vec![];
        if status == "open" || status == "closed" {
            where_sql = "WHERE t.status = ?".to_string();
            params.push(Value::from(status));
        }
        params.push(Value::from(limit as i64));

        let sql = format!(
            "
            SELECT
              t.topic_id,
              t.name,
              t.status,
              t.created_at,
              t.closed_at,
              t.close_reason,
              t.metadata_json,
              COUNT(m.message_id) AS message_count,
              COALESCE(MAX(m.seq), 0) AS last_seq
            FROM topics t
            LEFT JOIN messages m ON m.topic_id = t.topic_id
            {where_sql}
            GROUP BY t.topic_id
            ORDER BY t.created_at DESC
            LIMIT ?
            "
        );

        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(params.iter()), |row| {
                Ok(TopicCountRow {
                    topic_id: row.get("topic_id")?,
                    name: row.get("name")?,
                    status: row.get("status")?,
                    created_at: row.get("created_at")?,
                    closed_at: row.get("closed_at")?,
                    close_reason: row.get("close_reason")?,
                    metadata_json: row.get("metadata_json")?,
                    message_count: row.get("message_count")?,
                    last_seq: row.get("last_seq")?,
                })
            })
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            let dict = PyDict::new(py);
            dict.set_item("topic_id", row.topic_id)?;
            dict.set_item("name", row.name)?;
            dict.set_item("status", row.status)?;
            dict.set_item("created_at", row.created_at)?;
            dict.set_item("closed_at", row.closed_at)?;
            dict.set_item("close_reason", row.close_reason)?;
            dict.set_item("metadata_json", row.metadata_json)?;
            let counts = PyDict::new(py);
            counts.set_item("messages", row.message_count)?;
            counts.set_item("last_seq", row.last_seq)?;
            dict.set_item("counts", counts)?;
            out.append(dict)?;
        }
        Ok(out.into())
    }

    fn topic_close(
        &self,
        py: Python<'_>,
        topic_id: String,
        reason: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;
        let row = tx
            .query_row(
                "
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE topic_id = ?
                ",
                params![topic_id],
                topic_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        let existing = row.ok_or_else(|| TopicNotFoundError::new_err(topic_id.clone()))?;
        if existing.status == "closed" {
            let out = PyTuple::new(py, &[topic_to_dict(py, &existing), true.into_py_any(py)?])?;
            return Ok(out.into());
        }

        let closed_at = now();
        let close_reason = reason.filter(|r| !r.is_empty());
        tx.execute(
            "
            UPDATE topics
            SET status = 'closed', closed_at = ?, close_reason = ?
            WHERE topic_id = ?
            ",
            params![closed_at, close_reason, existing.topic_id],
        )
        .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;

        let updated = TopicRow {
            topic_id: existing.topic_id,
            name: existing.name,
            status: "closed".to_string(),
            created_at: existing.created_at,
            closed_at: Some(closed_at),
            close_reason,
            metadata_json: existing.metadata_json,
        };
        let out = PyTuple::new(py, &[topic_to_dict(py, &updated), false.into_py_any(py)?])?;
        Ok(out.into())
    }

    fn delete_topic(&self, topic_id: String) -> PyResult<bool> {
        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;
        let exists = tx
            .query_row(
                "SELECT topic_id FROM topics WHERE topic_id = ?",
                params![topic_id],
                |_r| Ok(true),
            )
            .optional()
            .map_err(map_db_error)?
            .unwrap_or(false);
        if !exists {
            tx.commit().map_err(map_db_error)?;
            return Ok(false);
        }

        tx.execute("DELETE FROM messages WHERE topic_id = ?", params![topic_id])
            .map_err(map_db_error)?;
        tx.execute("DELETE FROM cursors WHERE topic_id = ?", params![topic_id])
            .map_err(map_db_error)?;
        tx.execute(
            "DELETE FROM topic_seq WHERE topic_id = ?",
            params![topic_id],
        )
        .map_err(map_db_error)?;
        tx.execute("DELETE FROM topics WHERE topic_id = ?", params![topic_id])
            .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;
        Ok(true)
    }

    fn topic_rename(
        &self,
        py: Python<'_>,
        topic_id: String,
        new_name: String,
        rewrite_messages: bool,
    ) -> PyResult<Py<PyAny>> {
        if new_name.trim().is_empty() {
            return Err(PyValueError::new_err("new_name must be a non-empty string"));
        }
        let new_name = new_name.trim().to_string();

        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;
        let row = tx
            .query_row(
                "
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE topic_id = ?
                ",
                params![topic_id],
                topic_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        let existing = row.ok_or_else(|| TopicNotFoundError::new_err(topic_id.clone()))?;

        if existing.name == new_name {
            let out = PyTuple::new(
                py,
                &[
                    topic_to_dict(py, &existing),
                    true.into_py_any(py)?,
                    0_i64.into_py_any(py)?,
                ],
            )?;
            return Ok(out.into());
        }

        tx.execute(
            "UPDATE topics SET name = ? WHERE topic_id = ?",
            params![new_name, existing.topic_id],
        )
        .map_err(map_db_error)?;

        let mut rewritten = 0i64;
        if rewrite_messages {
            let old_name = existing.name.clone();
            if !old_name.is_empty() {
                rewritten = tx
                    .query_row(
                        "
                        SELECT COUNT(1) AS n
                        FROM messages
                        WHERE topic_id = ? AND instr(content_markdown, ?) > 0
                        ",
                        params![existing.topic_id, old_name],
                        |r| r.get::<_, i64>("n"),
                    )
                    .map_err(map_db_error)?;
                if rewritten > 0 {
                    tx.execute(
                        "
                        UPDATE messages
                        SET content_markdown = replace(content_markdown, ?, ?)
                        WHERE topic_id = ? AND instr(content_markdown, ?) > 0
                        ",
                        params![old_name, new_name, existing.topic_id, old_name],
                    )
                    .map_err(map_db_error)?;
                }
            }
        }

        tx.commit().map_err(map_db_error)?;
        let updated = TopicRow {
            topic_id: existing.topic_id,
            name: new_name,
            status: existing.status,
            created_at: existing.created_at,
            closed_at: existing.closed_at,
            close_reason: existing.close_reason,
            metadata_json: existing.metadata_json,
        };
        let out = PyTuple::new(
            py,
            &[
                topic_to_dict(py, &updated),
                false.into_py_any(py)?,
                rewritten.into_py_any(py)?,
            ],
        )?;
        Ok(out.into())
    }

    fn delete_message(&self, message_id: String) -> PyResult<i64> {
        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;
        let topic_id: Option<String> = tx
            .query_row(
                "SELECT topic_id FROM messages WHERE message_id = ?",
                params![message_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_db_error)?;
        let Some(topic_id) = topic_id else {
            tx.commit().map_err(map_db_error)?;
            return Ok(0);
        };

        let rows: Vec<String> = tx
            .prepare(
                "
                WITH RECURSIVE cascade(mid) AS (
                    SELECT message_id FROM messages WHERE message_id = ?
                    UNION ALL
                    SELECT m.message_id FROM messages m
                    JOIN cascade c ON m.reply_to = c.mid
                    WHERE m.topic_id = ?
                )
                SELECT mid FROM cascade
                ",
            )
            .map_err(map_db_error)?
            .query_map(params![message_id, topic_id], |r| r.get(0))
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        if rows.is_empty() {
            tx.commit().map_err(map_db_error)?;
            return Ok(0);
        }

        let placeholders = placeholders(rows.len());
        let delete_sql = format!("DELETE FROM messages WHERE message_id IN ({placeholders})");
        tx.execute(&delete_sql, params_from_iter(rows.iter()))
            .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;
        Ok(rows.len() as i64)
    }

    fn delete_messages_batch(
        &self,
        py: Python<'_>,
        topic_id: String,
        message_ids: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        if message_ids.is_empty() {
            return Ok(PyList::empty(py).into());
        }

        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;
        let id_placeholders = placeholders(message_ids.len());
        let sql = format!(
            "
            WITH RECURSIVE cascade(mid) AS (
                SELECT message_id
                FROM messages
                WHERE topic_id = ? AND message_id IN ({id_placeholders})
                UNION ALL
                SELECT m.message_id
                FROM messages m
                JOIN cascade c ON m.reply_to = c.mid
                WHERE m.topic_id = ?
            )
            SELECT mid FROM cascade
            "
        );

        let mut params_vec: Vec<Value> = vec![Value::from(topic_id.clone())];
        for mid in &message_ids {
            params_vec.push(Value::from(mid.clone()));
        }
        params_vec.push(Value::from(topic_id.clone()));

        let rows = {
            let mut stmt = tx.prepare(&sql).map_err(map_db_error)?;
            let result = stmt
                .query_map(params_from_iter(params_vec.iter()), |r| r.get(0))
                .map_err(map_db_error)?
                .collect::<Result<Vec<String>, _>>()
                .map_err(map_db_error)?;
            result
        };

        if rows.is_empty() {
            tx.commit().map_err(map_db_error)?;
            return Ok(PyList::empty(py).into());
        }

        let delete_placeholders = placeholders(rows.len());
        let delete_sql =
            format!("DELETE FROM messages WHERE message_id IN ({delete_placeholders})");
        tx.execute(&delete_sql, params_from_iter(rows.iter()))
            .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;

        let out = PyList::empty(py);
        for mid in rows {
            out.append(mid)?;
        }
        Ok(out.into())
    }

    fn topic_resolve(
        &self,
        py: Python<'_>,
        name: String,
        allow_closed: bool,
    ) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let row = conn
            .query_row(
                "
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE name = ? AND status = 'open'
                ORDER BY created_at DESC
                LIMIT 1
                ",
                params![name],
                topic_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        if let Some(topic) = row {
            return Ok(topic_to_dict(py, &topic));
        }
        if !allow_closed {
            return Err(TopicNotFoundError::new_err(name));
        }
        let row = conn
            .query_row(
                "
                SELECT topic_id, name, status, created_at, closed_at, close_reason, metadata_json
                FROM topics
                WHERE name = ? AND status = 'closed'
                ORDER BY created_at DESC
                LIMIT 1
                ",
                params![name],
                topic_row_from,
            )
            .optional()
            .map_err(map_db_error)?;
        let topic = row.ok_or_else(|| TopicNotFoundError::new_err(name))?;
        Ok(topic_to_dict(py, &topic))
    }

    #[allow(clippy::too_many_arguments)]
    fn sync_once(
        &self,
        py: Python<'_>,
        topic_id: String,
        agent_name: String,
        outbox: Vec<OutboxItem>,
        max_items: usize,
        include_self: bool,
        auto_advance: bool,
        ack_through: Option<i64>,
        now_override: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        let updated_at = now_override.unwrap_or_else(now);
        let created_at = updated_at;

        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;

        let topic_status: Option<String> = tx
            .query_row(
                "SELECT status FROM topics WHERE topic_id = ?",
                params![topic_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_db_error)?;
        let Some(status) = topic_status else {
            return Err(TopicNotFoundError::new_err(topic_id));
        };
        if !outbox.is_empty() && status != "open" {
            return Err(TopicClosedError::new_err("topic closed"));
        }

        tx.execute(
            "
            INSERT OR IGNORE INTO cursors(topic_id, agent_name, last_seq, updated_at)
            VALUES (?, ?, 0, ?)
            ",
            params![topic_id, agent_name, updated_at],
        )
        .map_err(map_db_error)?;

        let cursor_row = tx
            .query_row(
                "
                SELECT topic_id, agent_name, last_seq, updated_at
                FROM cursors
                WHERE topic_id = ? AND agent_name = ?
                ",
                params![topic_id, agent_name],
                cursor_row_from,
            )
            .map_err(map_db_error)?;
        let mut cursor = cursor_row;

        tx.execute(
            "
            INSERT OR IGNORE INTO topic_seq(topic_id, next_seq, updated_at)
            VALUES (?, 1, ?)
            ",
            params![topic_id, updated_at],
        )
        .map_err(map_db_error)?;

        let next_seq: i64 = tx
            .query_row(
                "SELECT next_seq FROM topic_seq WHERE topic_id = ?",
                params![topic_id],
                |r| r.get(0),
            )
            .map_err(map_db_error)?;
        let mut next_seq = next_seq;

        let mut sent: Vec<(MessageRow, bool)> = Vec::new();
        for item in outbox {
            if let Some(client_message_id) = item.client_message_id.clone() {
                let existing = tx
                    .query_row(
                        "
                        SELECT
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        FROM messages
                        WHERE topic_id = ? AND sender = ? AND client_message_id = ?
                        ",
                        params![topic_id, agent_name, client_message_id],
                        message_row_from,
                    )
                    .optional()
                    .map_err(map_db_error)?;
                if let Some(row) = existing {
                    sent.push((row, true));
                    continue;
                }
            }

            let message_id = new_id();
            let seq = next_seq;
            next_seq += 1;

            let metadata_json = item.metadata_json.clone();

            let res = tx.execute(
                "
                INSERT INTO messages(
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ",
                params![
                    message_id,
                    topic_id,
                    seq,
                    agent_name,
                    item.message_type,
                    item.reply_to,
                    item.content_markdown,
                    metadata_json,
                    item.client_message_id,
                    created_at
                ],
            );
            if let Err(err) = res {
                if let rusqlite::Error::SqliteFailure(_, _) = err {
                    if item.client_message_id.is_some() {
                        let existing = tx
                            .query_row(
                                "
                                SELECT
                                  message_id, topic_id, seq, sender, message_type, reply_to,
                                  content_markdown, metadata_json, client_message_id, created_at
                                FROM messages
                                WHERE topic_id = ? AND sender = ? AND client_message_id = ?
                                ",
                                params![topic_id, agent_name, item.client_message_id],
                                message_row_from,
                            )
                            .optional()
                            .map_err(map_db_error)?;
                        if let Some(row) = existing {
                            sent.push((row, true));
                            continue;
                        }
                    }
                }
                return Err(map_db_error(err));
            }

            sent.push((
                MessageRow {
                    message_id,
                    topic_id: topic_id.clone(),
                    seq,
                    sender: agent_name.clone(),
                    message_type: item.message_type,
                    reply_to: item.reply_to,
                    content_markdown: item.content_markdown,
                    metadata_json,
                    client_message_id: item.client_message_id,
                    created_at,
                },
                false,
            ));
        }

        tx.execute(
            "
            UPDATE topic_seq
            SET next_seq = ?, updated_at = ?
            WHERE topic_id = ?
            ",
            params![next_seq, updated_at, topic_id],
        )
        .map_err(map_db_error)?;

        if !auto_advance {
            if let Some(ack) = ack_through {
                if ack < 0 {
                    return Err(PyValueError::new_err("ack_through must be >= 0"));
                }
                let max_seq = next_seq - 1;
                if ack > max_seq {
                    return Err(PyValueError::new_err(
                        "ack_through exceeds latest message seq",
                    ));
                }
                if ack > cursor.last_seq {
                    tx.execute(
                        "
                        UPDATE cursors
                        SET last_seq = ?, updated_at = ?
                        WHERE topic_id = ? AND agent_name = ?
                        ",
                        params![ack, updated_at, topic_id, agent_name],
                    )
                    .map_err(map_db_error)?;
                    cursor.last_seq = ack;
                    cursor.updated_at = updated_at;
                }
            }
        }

        let mut where_sql = "topic_id = ? AND seq > ?".to_string();
        let mut params_vec: Vec<Value> =
            vec![Value::from(topic_id.clone()), Value::from(cursor.last_seq)];
        if !include_self {
            where_sql.push_str(" AND sender <> ?");
            params_vec.push(Value::from(agent_name.clone()));
        }
        params_vec.push(Value::from((max_items + 1) as i64));

        let sql = format!(
            "
            SELECT
              message_id, topic_id, seq, sender, message_type, reply_to,
              content_markdown, metadata_json, client_message_id, created_at
            FROM messages
            WHERE {where_sql}
            ORDER BY seq ASC
            LIMIT ?
            "
        );
        let (received_slice, has_more) = {
            let mut stmt = tx.prepare(&sql).map_err(map_db_error)?;
            let rows: Vec<MessageRow> = stmt
                .query_map(params_from_iter(params_vec.iter()), message_row_from)
                .map_err(map_db_error)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_db_error)?;
            let has_more = rows.len() > max_items;
            (
                rows.into_iter().take(max_items).collect::<Vec<_>>(),
                has_more,
            )
        };

        if auto_advance && !received_slice.is_empty() {
            let new_last_seq = received_slice
                .iter()
                .map(|m| m.seq)
                .max()
                .unwrap_or(cursor.last_seq);
            if new_last_seq > cursor.last_seq {
                tx.execute(
                    "
                    UPDATE cursors
                    SET last_seq = ?, updated_at = ?
                    WHERE topic_id = ? AND agent_name = ?
                    ",
                    params![new_last_seq, updated_at, topic_id, agent_name],
                )
                .map_err(map_db_error)?;
                cursor.last_seq = new_last_seq;
                cursor.updated_at = updated_at;
            }
        }

        if (cursor.updated_at - updated_at).abs() > f64::EPSILON {
            tx.execute(
                "
                UPDATE cursors
                SET updated_at = ?
                WHERE topic_id = ? AND agent_name = ?
                ",
                params![updated_at, topic_id, agent_name],
            )
            .map_err(map_db_error)?;
            cursor.updated_at = updated_at;
        }

        tx.commit().map_err(map_db_error)?;

        let sent_py = PyList::empty(py);
        for (msg, dup) in sent {
            let tup = PyTuple::new(py, &[message_to_dict(py, &msg), dup.into_py_any(py)?])?;
            sent_py.append(tup)?;
        }

        let received_py = PyList::empty(py);
        for msg in received_slice {
            received_py.append(message_to_dict(py, &msg))?;
        }

        let cursor_py = cursor_to_dict(py, &cursor);
        let out = PyTuple::new(
            py,
            &[
                sent_py.into(),
                received_py.into(),
                cursor_py,
                has_more.into_py_any(py)?,
            ],
        )?;
        Ok(out.into())
    }

    fn cursor_set(
        &self,
        py: Python<'_>,
        topic_id: String,
        agent_name: String,
        last_seq: i64,
    ) -> PyResult<Py<PyAny>> {
        if last_seq < 0 {
            return Err(PyValueError::new_err("last_seq must be >= 0"));
        }
        let updated_at = now();
        let mut conn = self.connect()?;
        let tx = conn.transaction().map_err(map_db_error)?;

        let exists = tx
            .query_row(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                params![topic_id],
                |_r| Ok(true),
            )
            .optional()
            .map_err(map_db_error)?
            .unwrap_or(false);
        if !exists {
            return Err(TopicNotFoundError::new_err(topic_id));
        }

        tx.execute(
            "
            INSERT OR IGNORE INTO cursors(topic_id, agent_name, last_seq, updated_at)
            VALUES (?, ?, 0, ?)
            ",
            params![topic_id, agent_name, updated_at],
        )
        .map_err(map_db_error)?;

        let seq_row: Option<i64> = tx
            .query_row(
                "SELECT next_seq FROM topic_seq WHERE topic_id = ?",
                params![topic_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_db_error)?;
        let max_seq = seq_row.map(|n| (n - 1).max(0)).unwrap_or(0);
        if last_seq > max_seq {
            return Err(PyValueError::new_err("last_seq exceeds latest message seq"));
        }

        tx.execute(
            "
            UPDATE cursors
            SET last_seq = ?, updated_at = ?
            WHERE topic_id = ? AND agent_name = ?
            ",
            params![last_seq, updated_at, topic_id, agent_name],
        )
        .map_err(map_db_error)?;
        tx.commit().map_err(map_db_error)?;

        let cursor = CursorRow {
            topic_id,
            agent_name,
            last_seq,
            updated_at,
        };
        Ok(cursor_to_dict(py, &cursor))
    }

    #[pyo3(signature = (topic_id, window_seconds=300, limit=200, now_override=None))]
    fn get_presence(
        &self,
        py: Python<'_>,
        topic_id: String,
        window_seconds: i64,
        limit: usize,
        now_override: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        if window_seconds <= 0 {
            return Err(PyValueError::new_err("window_seconds must be > 0"));
        }
        if limit == 0 {
            return Err(PyValueError::new_err("limit must be > 0"));
        }
        let cutoff = now_override.unwrap_or_else(now) - window_seconds as f64;
        let conn = self.connect()?;
        let exists = conn
            .query_row(
                "SELECT 1 FROM topics WHERE topic_id = ?",
                params![topic_id],
                |_r| Ok(true),
            )
            .optional()
            .map_err(map_db_error)?
            .unwrap_or(false);
        if !exists {
            return Err(TopicNotFoundError::new_err(topic_id));
        }

        let mut stmt = conn
            .prepare(
                "
                SELECT topic_id, agent_name, last_seq, updated_at
                FROM cursors
                WHERE topic_id = ? AND updated_at >= ?
                ORDER BY updated_at DESC
                LIMIT ?
                ",
            )
            .map_err(map_db_error)?;
        let rows = stmt
            .query_map(params![topic_id, cutoff, limit as i64], cursor_row_from)
            .map_err(map_db_error)?;

        let out = PyList::empty(py);
        for row in rows {
            out.append(cursor_to_dict(py, &row.map_err(map_db_error)?))?;
        }
        Ok(out.into())
    }

    #[pyo3(signature = (topic_id, after_seq=0, before_seq=None, limit=100))]
    fn get_messages(
        &self,
        py: Python<'_>,
        topic_id: String,
        after_seq: i64,
        before_seq: Option<i64>,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;

        let rows = if let Some(before) = before_seq {
            if after_seq == 0 {
                let mut stmt = conn
                    .prepare(
                        "
                        SELECT
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        FROM messages
                        WHERE topic_id = ? AND seq < ?
                        ORDER BY seq DESC
                        LIMIT ?
                        ",
                    )
                    .map_err(map_db_error)?;
                let mut rows = stmt
                    .query_map(params![topic_id, before, limit as i64], message_row_from)
                    .map_err(map_db_error)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(map_db_error)?;
                rows.reverse();
                rows
            } else {
                let mut stmt = conn
                    .prepare(
                        "
                        SELECT
                          message_id, topic_id, seq, sender, message_type, reply_to,
                          content_markdown, metadata_json, client_message_id, created_at
                        FROM messages
                        WHERE topic_id = ? AND seq > ? AND seq < ?
                        ORDER BY seq ASC
                        LIMIT ?
                        ",
                    )
                    .map_err(map_db_error)?;
                let rows = stmt
                    .query_map(
                        params![topic_id, after_seq, before, limit as i64],
                        message_row_from,
                    )
                    .map_err(map_db_error)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(map_db_error)?;
                rows
            }
        } else {
            let mut stmt = conn
                .prepare(
                    "
                    SELECT
                      message_id, topic_id, seq, sender, message_type, reply_to,
                      content_markdown, metadata_json, client_message_id, created_at
                    FROM messages
                    WHERE topic_id = ? AND seq > ?
                    ORDER BY seq ASC
                    LIMIT ?
                    ",
                )
                .map_err(map_db_error)?;
            let rows = stmt
                .query_map(params![topic_id, after_seq, limit as i64], message_row_from)
                .map_err(map_db_error)?
                .collect::<Result<Vec<_>, _>>()
                .map_err(map_db_error)?;
            rows
        };

        let out = PyList::empty(py);
        for msg in rows {
            out.append(message_to_dict(py, &msg))?;
        }
        Ok(out.into())
    }

    fn get_latest_messages(
        &self,
        py: Python<'_>,
        topic_id: String,
        limit: usize,
    ) -> PyResult<Py<PyAny>> {
        let conn = self.connect()?;
        let mut stmt = conn
            .prepare(
                "
                SELECT
                  message_id, topic_id, seq, sender, message_type, reply_to,
                  content_markdown, metadata_json, client_message_id, created_at
                FROM messages
                WHERE topic_id = ?
                ORDER BY seq DESC
                LIMIT ?
                ",
            )
            .map_err(map_db_error)?;
        let mut rows = stmt
            .query_map(params![topic_id, limit as i64], message_row_from)
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;
        rows.reverse();
        let out = PyList::empty(py);
        for msg in rows {
            out.append(message_to_dict(py, &msg))?;
        }
        Ok(out.into())
    }

    fn get_senders_by_message_ids(
        &self,
        py: Python<'_>,
        message_ids: Vec<String>,
    ) -> PyResult<Py<PyAny>> {
        if message_ids.is_empty() {
            return Ok(PyDict::new(py).into());
        }
        let conn = self.connect()?;
        let placeholders = placeholders(message_ids.len());
        let sql = format!(
            "
            SELECT message_id, sender
            FROM messages
            WHERE message_id IN ({placeholders})
            "
        );
        let mut stmt = conn.prepare(&sql).map_err(map_db_error)?;
        let rows = stmt
            .query_map(params_from_iter(message_ids.iter()), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(map_db_error)?;
        let out = PyDict::new(py);
        for row in rows {
            let (mid, sender) = row.map_err(map_db_error)?;
            out.set_item(mid, sender)?;
        }
        Ok(out.into())
    }
}

#[pymodule]
fn _core(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CoreDb>()?;
    module.add_function(wrap_pyfunction!(embed_texts, module)?)?;
    module.add("DBBusyError", py.get_type::<DBBusyError>())?;
    module.add("SchemaMismatchError", py.get_type::<SchemaMismatchError>())?;
    module.add("TopicNotFoundError", py.get_type::<TopicNotFoundError>())?;
    module.add("TopicClosedError", py.get_type::<TopicClosedError>())?;
    module.add("TopicMismatchError", py.get_type::<TopicMismatchError>())?;
    module.add("SCHEMA_VERSION", SCHEMA_VERSION)?;
    Ok(())
}

fn resolve_db_path(path: Option<String>) -> String {
    let raw = path
        .or_else(|| std::env::var("AGENT_BUS_DB").ok())
        .unwrap_or_else(default_db_path);
    if raw == ":memory:" {
        return raw;
    }
    expand_home(&raw).unwrap_or(raw)
}

fn default_db_path() -> String {
    let mut base = home_dir().unwrap_or_else(|| PathBuf::from("."));
    base.push(".agent_bus");
    base.push("agent_bus.sqlite");
    base.to_string_lossy().to_string()
}

fn home_dir() -> Option<PathBuf> {
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    if let Ok(home) = std::env::var("USERPROFILE") {
        if !home.is_empty() {
            return Some(PathBuf::from(home));
        }
    }
    let drive = std::env::var("HOMEDRIVE").ok();
    let path = std::env::var("HOMEPATH").ok();
    if let (Some(d), Some(p)) = (drive, path) {
        return Some(PathBuf::from(format!("{d}{p}")));
    }
    None
}

fn expand_home(path: &str) -> Option<String> {
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir().map(|h| h.join(rest).to_string_lossy().to_string());
    }
    if let Some(rest) = path.strip_prefix("~\\") {
        return home_dir().map(|h| h.join(rest).to_string_lossy().to_string());
    }
    None
}

impl CoreDb {
    fn connect(&self) -> PyResult<Connection> {
        ensure_parent_dir(&self.path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let conn = Connection::open(&self.path).map_err(map_db_error)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(map_db_error)?;
        conn.busy_timeout(Duration::from_millis(2000))
            .map_err(map_db_error)?;
        self.ensure_schema(&conn)?;
        Ok(conn)
    }

    fn ensure_schema(&self, conn: &Connection) -> PyResult<()> {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .map_err(map_db_error)?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(map_db_error)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(map_db_error)?;

        if !rows.iter().any(|n| n == "meta") {
            if !rows.is_empty() {
                return Err(SchemaMismatchError::new_err(
                    "Database schema is outdated (missing schema version). Wipe it with `agent-bus cli db wipe --yes` or delete the file at $AGENT_BUS_DB.",
                ));
            }
            conn.execute_batch(&format!(
                "
                CREATE TABLE meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                INSERT INTO meta(key, value)
                VALUES ('schema_version', '{SCHEMA_VERSION}');
                "
            ))
            .map_err(map_db_error)?;
        } else {
            let version: Option<String> = conn
                .query_row(
                    "SELECT value FROM meta WHERE key = 'schema_version'",
                    [],
                    |r| r.get(0),
                )
                .optional()
                .map_err(map_db_error)?;
            if version.as_deref() != Some(SCHEMA_VERSION) {
                return Err(SchemaMismatchError::new_err(
                    "Database schema version mismatch. Wipe it with `agent-bus cli db wipe --yes` or delete the file at $AGENT_BUS_DB.",
                ));
            }
        }

        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS topics (
              topic_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              created_at REAL NOT NULL,
              status TEXT NOT NULL,
              closed_at REAL NULL,
              close_reason TEXT NULL,
              metadata_json TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS topic_seq (
              topic_id TEXT PRIMARY KEY,
              next_seq INTEGER NOT NULL,
              updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              topic_id TEXT NOT NULL,
              seq INTEGER NOT NULL,
              sender TEXT NOT NULL,
              message_type TEXT NOT NULL,
              reply_to TEXT NULL,
              content_markdown TEXT NOT NULL,
              metadata_json TEXT NULL,
              client_message_id TEXT NULL,
              created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cursors (
              topic_id TEXT NOT NULL,
              agent_name TEXT NOT NULL,
              last_seq INTEGER NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY(topic_id, agent_name)
            );

            CREATE INDEX IF NOT EXISTS idx_topics_name_status_created_at
              ON topics(name, status, created_at);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_topic_seq_unique
              ON messages(topic_id, seq);

            CREATE INDEX IF NOT EXISTS idx_messages_topic_seq
              ON messages(topic_id, seq);

            CREATE INDEX IF NOT EXISTS idx_messages_topic_reply_to
              ON messages(topic_id, reply_to);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_topic_sender_client_id_unique
              ON messages(topic_id, sender, client_message_id)
              WHERE client_message_id IS NOT NULL;
            ",
        )
        .map_err(map_db_error)?;

        self.ensure_search_schema(conn)?;
        self.ensure_embeddings_schema(conn)?;

        Ok(())
    }

    fn ensure_search_schema(&self, conn: &Connection) -> PyResult<()> {
        let mut available = true;
        let result = conn.execute_batch(
            "
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
              content_markdown,
              message_id UNINDEXED,
              topic_id UNINDEXED,
              tokenize='unicode61'
            );
            ",
        );
        if let Err(err) = result {
            if is_fts_unavailable(&err) {
                available = false;
            } else {
                return Err(map_db_error(err));
            }
        }

        let mut guard = self
            .fts_available
            .lock()
            .map_err(|_| PyRuntimeError::new_err("failed to update FTS availability state"))?;
        *guard = Some(available);

        if !available {
            return Ok(());
        }

        conn.execute_batch(
            "
            DROP TRIGGER IF EXISTS messages_fts_ai;
            DROP TRIGGER IF EXISTS messages_fts_ad;
            DROP TRIGGER IF EXISTS messages_fts_au;

            CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
              INSERT INTO messages_fts(rowid, content_markdown, message_id, topic_id)
              VALUES (new.rowid, new.content_markdown, new.message_id, new.topic_id);
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
              DELETE FROM messages_fts WHERE rowid = old.rowid;
            END;

            CREATE TRIGGER IF NOT EXISTS messages_fts_au AFTER UPDATE ON messages BEGIN
              DELETE FROM messages_fts WHERE rowid = old.rowid;
              INSERT INTO messages_fts(rowid, content_markdown, message_id, topic_id)
              VALUES (new.rowid, new.content_markdown, new.message_id, new.topic_id);
            END;
            ",
        )
        .map_err(map_db_error)?;

        let existing: Option<i64> = conn
            .query_row(
                "SELECT 1 FROM meta WHERE key = 'fts_backfill_v1' AND value = '1'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(map_db_error)?;
        if existing.is_none() {
            conn.execute_batch(
                "
                INSERT INTO messages_fts(rowid, content_markdown, message_id, topic_id)
                SELECT rowid, content_markdown, message_id, topic_id
                FROM messages;
                ",
            )
            .map_err(map_db_error)?;
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('fts_backfill_v1', '1')",
                [],
            )
            .map_err(map_db_error)?;
        }
        Ok(())
    }

    fn ensure_embeddings_schema(&self, conn: &Connection) -> PyResult<()> {
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS message_embedding_state (
              message_id TEXT NOT NULL,
              model TEXT NOT NULL,
              content_hash TEXT NOT NULL,
              chunk_size INTEGER NOT NULL,
              chunk_overlap INTEGER NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY(message_id, model)
            );

            CREATE TABLE IF NOT EXISTS chunk_embeddings (
              message_id TEXT NOT NULL,
              model TEXT NOT NULL,
              topic_id TEXT NOT NULL,
              chunk_index INTEGER NOT NULL,
              start_char INTEGER NOT NULL,
              end_char INTEGER NOT NULL,
              dims INTEGER NOT NULL,
              vector BLOB NOT NULL,
              text_hash TEXT NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY(message_id, model, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_topic_model
              ON chunk_embeddings(topic_id, model);

            CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
              ON chunk_embeddings(model);

            CREATE TABLE IF NOT EXISTS embedding_jobs (
              message_id TEXT NOT NULL,
              model TEXT NOT NULL,
              topic_id TEXT NOT NULL,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL,
              locked_by TEXT NULL,
              locked_at REAL NULL,
              last_error TEXT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              PRIMARY KEY(message_id, model)
            );

            CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status_model_updated
              ON embedding_jobs(status, model, updated_at);

            DROP TRIGGER IF EXISTS messages_embeddings_ad;
            DROP TRIGGER IF EXISTS messages_embeddings_au;

            CREATE TRIGGER IF NOT EXISTS messages_embeddings_ad AFTER DELETE ON messages BEGIN
              DELETE FROM chunk_embeddings WHERE message_id = old.message_id;
              DELETE FROM message_embedding_state WHERE message_id = old.message_id;
              DELETE FROM embedding_jobs WHERE message_id = old.message_id;
            END;

            CREATE TRIGGER IF NOT EXISTS messages_embeddings_au AFTER UPDATE OF content_markdown ON messages BEGIN
              DELETE FROM chunk_embeddings WHERE message_id = old.message_id;
              DELETE FROM message_embedding_state WHERE message_id = old.message_id;
              UPDATE embedding_jobs
              SET status = 'pending',
                  attempts = 0,
                  locked_by = NULL,
                  locked_at = NULL,
                  last_error = NULL
              WHERE message_id = old.message_id;
            END;
            ",
        )
        .map_err(map_db_error)?;
        Ok(())
    }
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs_f64()
}

fn new_id() -> String {
    Uuid::new_v4().simple().to_string()[0..10].to_string()
}

fn ensure_parent_dir(path: &str) -> std::io::Result<()> {
    if path == ":memory:" {
        return Ok(());
    }
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn placeholders(count: usize) -> String {
    (0..count).map(|_| "?").collect::<Vec<_>>().join(",")
}

fn map_db_error(err: rusqlite::Error) -> PyErr {
    let msg = err.to_string();
    let lowered = msg.to_lowercase();
    if lowered.contains("locked") || lowered.contains("busy") {
        return DBBusyError::new_err(msg);
    }
    PyRuntimeError::new_err(msg)
}

fn is_fts_unavailable(err: &rusqlite::Error) -> bool {
    if let rusqlite::Error::SqliteFailure(_, Some(msg)) = err {
        let lowered = msg.to_lowercase();
        return lowered.contains("fts5") || lowered.contains("no such module");
    }
    false
}

fn topic_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<TopicRow> {
    Ok(TopicRow {
        topic_id: row.get("topic_id")?,
        name: row.get("name")?,
        status: row.get("status")?,
        created_at: row.get("created_at")?,
        closed_at: row.get("closed_at")?,
        close_reason: row.get("close_reason")?,
        metadata_json: row.get("metadata_json")?,
    })
}

fn message_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        message_id: row.get("message_id")?,
        topic_id: row.get("topic_id")?,
        seq: row.get("seq")?,
        sender: row.get("sender")?,
        message_type: row.get("message_type")?,
        reply_to: row.get("reply_to")?,
        content_markdown: row.get("content_markdown")?,
        metadata_json: row.get("metadata_json")?,
        client_message_id: row.get("client_message_id")?,
        created_at: row.get("created_at")?,
    })
}

fn cursor_row_from(row: &rusqlite::Row<'_>) -> rusqlite::Result<CursorRow> {
    Ok(CursorRow {
        topic_id: row.get("topic_id")?,
        agent_name: row.get("agent_name")?,
        last_seq: row.get("last_seq")?,
        updated_at: row.get("updated_at")?,
    })
}

fn topic_to_dict(py: Python<'_>, topic: &TopicRow) -> Py<PyAny> {
    let dict = PyDict::new(py);
    dict.set_item("topic_id", &topic.topic_id).unwrap();
    dict.set_item("name", &topic.name).unwrap();
    dict.set_item("status", &topic.status).unwrap();
    dict.set_item("created_at", topic.created_at).unwrap();
    dict.set_item("closed_at", topic.closed_at).unwrap();
    dict.set_item("close_reason", &topic.close_reason).unwrap();
    dict.set_item("metadata_json", &topic.metadata_json)
        .unwrap();
    dict.into()
}

fn message_to_dict(py: Python<'_>, msg: &MessageRow) -> Py<PyAny> {
    let dict = PyDict::new(py);
    dict.set_item("message_id", &msg.message_id).unwrap();
    dict.set_item("topic_id", &msg.topic_id).unwrap();
    dict.set_item("seq", msg.seq).unwrap();
    dict.set_item("sender", &msg.sender).unwrap();
    dict.set_item("message_type", &msg.message_type).unwrap();
    dict.set_item("reply_to", &msg.reply_to).unwrap();
    dict.set_item("content_markdown", &msg.content_markdown)
        .unwrap();
    dict.set_item("metadata_json", &msg.metadata_json).unwrap();
    dict.set_item("client_message_id", &msg.client_message_id)
        .unwrap();
    dict.set_item("created_at", msg.created_at).unwrap();
    dict.into()
}

fn cursor_to_dict(py: Python<'_>, cursor: &CursorRow) -> Py<PyAny> {
    let dict = PyDict::new(py);
    dict.set_item("topic_id", &cursor.topic_id).unwrap();
    dict.set_item("agent_name", &cursor.agent_name).unwrap();
    dict.set_item("last_seq", cursor.last_seq).unwrap();
    dict.set_item("updated_at", cursor.updated_at).unwrap();
    dict.into()
}
