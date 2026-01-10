//! Code execution tool for running code in a sandbox.
//!
//! This module provides a configurable code execution tool that can
//! execute code in various programming languages.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::time::Duration;

use crate::{
    definition::ToolDefinition,
    errors::ToolError,
    return_types::{ToolResult, ToolReturn},
    schema::SchemaBuilder,
    tool::Tool,
    RunContext,
};

/// Configuration for the code execution tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {
    /// Maximum execution time.
    #[serde(with = "humantime_serde")]
    pub timeout: Duration,
    /// Maximum output size in bytes.
    pub max_output_size: usize,
    /// Allowed languages.
    pub allowed_languages: Vec<ProgrammingLanguage>,
    /// Whether to capture stderr.
    pub capture_stderr: bool,
    /// Working directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub working_dir: Option<String>,
    /// Environment variables.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub env_vars: Vec<(String, String)>,
}

impl Default for CodeExecutionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_output_size: 1024 * 1024, // 1MB
            allowed_languages: vec![ProgrammingLanguage::Python, ProgrammingLanguage::JavaScript],
            capture_stderr: true,
            working_dir: None,
            env_vars: Vec::new(),
        }
    }
}

impl CodeExecutionConfig {
    /// Create a new config with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set timeout in seconds.
    #[must_use]
    pub fn timeout_secs(self, secs: u64) -> Self {
        self.timeout(Duration::from_secs(secs))
    }

    /// Set max output size.
    #[must_use]
    pub fn max_output_size(mut self, size: usize) -> Self {
        self.max_output_size = size;
        self
    }

    /// Set allowed languages.
    #[must_use]
    pub fn allowed_languages(mut self, langs: Vec<ProgrammingLanguage>) -> Self {
        self.allowed_languages = langs;
        self
    }

    /// Add an allowed language.
    #[must_use]
    pub fn allow_language(mut self, lang: ProgrammingLanguage) -> Self {
        if !self.allowed_languages.contains(&lang) {
            self.allowed_languages.push(lang);
        }
        self
    }

    /// Set capture stderr.
    #[must_use]
    pub fn capture_stderr(mut self, capture: bool) -> Self {
        self.capture_stderr = capture;
        self
    }

    /// Add an environment variable.
    #[must_use]
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.push((key.into(), value.into()));
        self
    }
}

/// Programming languages supported for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProgrammingLanguage {
    /// Python 3.
    Python,
    /// JavaScript (Node.js).
    JavaScript,
    /// TypeScript.
    TypeScript,
    /// Ruby.
    Ruby,
    /// Go.
    Go,
    /// Rust.
    Rust,
    /// Shell/Bash.
    Shell,
    /// SQL.
    Sql,
}

impl ProgrammingLanguage {
    /// Get the language name as a string.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::JavaScript => "javascript",
            Self::TypeScript => "typescript",
            Self::Ruby => "ruby",
            Self::Go => "go",
            Self::Rust => "rust",
            Self::Shell => "shell",
            Self::Sql => "sql",
        }
    }

    /// Parse from a string.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "python" | "py" => Some(Self::Python),
            "javascript" | "js" => Some(Self::JavaScript),
            "typescript" | "ts" => Some(Self::TypeScript),
            "ruby" | "rb" => Some(Self::Ruby),
            "go" | "golang" => Some(Self::Go),
            "rust" | "rs" => Some(Self::Rust),
            "shell" | "bash" | "sh" => Some(Self::Shell),
            "sql" => Some(Self::Sql),
            _ => None,
        }
    }

    /// Get all language names for schema enum.
    #[must_use]
    pub fn all_names() -> &'static [&'static str] {
        &[
            "python",
            "javascript",
            "typescript",
            "ruby",
            "go",
            "rust",
            "shell",
            "sql",
        ]
    }
}

impl std::fmt::Display for ProgrammingLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Result of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Standard output.
    pub stdout: String,
    /// Standard error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr: Option<String>,
    /// Exit code.
    pub exit_code: i32,
    /// Execution time in milliseconds.
    pub execution_time_ms: u64,
    /// Whether execution timed out.
    pub timed_out: bool,
}

impl ExecutionResult {
    /// Check if execution was successful.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }
}

/// Code execution tool.
///
/// This tool allows agents to execute code in a sandboxed environment.
/// It requires integration with an external code execution service.
///
/// # Safety
///
/// Code execution is inherently dangerous. This tool should:
/// - Always run in a sandboxed environment
/// - Have strict resource limits
/// - Only be used with trusted agents
///
/// # Example
///
/// ```ignore
/// use serdes_ai_tools::builtin::{CodeExecutionTool, CodeExecutionConfig, ProgrammingLanguage};
///
/// let tool = CodeExecutionTool::with_config(
///     CodeExecutionConfig::new()
///         .timeout_secs(10)
///         .allowed_languages(vec![ProgrammingLanguage::Python])
/// );
/// ```
pub struct CodeExecutionTool {
    config: CodeExecutionConfig,
}

impl CodeExecutionTool {
    /// Create a new code execution tool with default config.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CodeExecutionConfig::default(),
        }
    }

    /// Create with a specific config.
    #[must_use]
    pub fn with_config(config: CodeExecutionConfig) -> Self {
        Self { config }
    }

    /// Get the tool schema.
    fn schema(&self) -> JsonValue {
        let lang_names: Vec<&str> = self
            .config
            .allowed_languages
            .iter()
            .map(|l| l.as_str())
            .collect();

        SchemaBuilder::new()
            .enum_values(
                "language",
                "The programming language to execute",
                &lang_names,
                true,
            )
            .string("code", "The code to execute", true)
            .string(
                "stdin",
                "Optional input to provide to the program via stdin",
                false,
            )
            .build()
            .expect("SchemaBuilder JSON serialization failed")
    }

    /// Execute code (stub - integrate with actual sandbox).
    async fn execute(
        &self,
        language: ProgrammingLanguage,
        code: &str,
        _stdin: Option<&str>,
    ) -> ExecutionResult {
        // This is a stub implementation.
        // In a real implementation, you would:
        // 1. Send the code to a sandbox service (e.g., Docker, Firecracker, etc.)
        // 2. Execute with proper resource limits
        // 3. Capture output and handle timeouts

        ExecutionResult {
            stdout: format!(
                "[Placeholder] Would execute {} code:\n{}\n\n\
                 Integrate with a sandbox service for real execution.",
                language, code
            ),
            stderr: None,
            exit_code: 0,
            execution_time_ms: 0,
            timed_out: false,
        }
    }
}

impl Default for CodeExecutionTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<Deps: Send + Sync> Tool<Deps> for CodeExecutionTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition::new("code_execution", "Execute code in a sandboxed environment")
            .with_parameters(self.schema())
    }

    async fn call(&self, _ctx: &RunContext<Deps>, args: JsonValue) -> ToolResult {
        let language_str = args
            .get("language")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ToolError::validation_error(
                    "code_execution",
                    Some("language".to_string()),
                    "Missing 'language' field",
                )
            })?;

        let language = ProgrammingLanguage::from_str(language_str).ok_or_else(|| {
            ToolError::validation_error(
                "code_execution",
                Some("language".to_string()),
                format!("Unknown language: {}", language_str),
            )
        })?;

        if !self.config.allowed_languages.contains(&language) {
            return Err(ToolError::validation_error(
                "code_execution",
                Some("language".to_string()),
                format!(
                    "Language '{}' is not allowed. Allowed: {:?}",
                    language, self.config.allowed_languages
                ),
            ));
        }

        let code = args.get("code").and_then(|v| v.as_str()).ok_or_else(|| {
            ToolError::validation_error(
                "code_execution",
                Some("code".to_string()),
                "Missing 'code' field",
            )
        })?;

        if code.trim().is_empty() {
            return Err(ToolError::validation_error(
                "code_execution",
                Some("code".to_string()),
                "Code cannot be empty",
            ));
        }

        let stdin = args.get("stdin").and_then(|v| v.as_str());

        let result = self.execute(language, code, stdin).await;

        let output = serde_json::json!({
            "success": result.is_success(),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "execution_time_ms": result.execution_time_ms,
            "timed_out": result.timed_out
        });

        Ok(ToolReturn::json(output))
    }

    fn max_retries(&self) -> Option<u32> {
        Some(1)
    }
}

impl std::fmt::Debug for CodeExecutionTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeExecutionTool")
            .field("config", &self.config)
            .finish()
    }
}

/// Trait for code execution providers.
#[allow(async_fn_in_trait)]
pub trait CodeExecutor: Send + Sync {
    /// Execute code in a sandbox.
    async fn execute(
        &self,
        language: ProgrammingLanguage,
        code: &str,
        stdin: Option<&str>,
        config: &CodeExecutionConfig,
    ) -> Result<ExecutionResult, ToolError>;
}

/// Serde helper for Duration.
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_execution_config() {
        let config = CodeExecutionConfig::new()
            .timeout_secs(10)
            .max_output_size(1024)
            .allowed_languages(vec![ProgrammingLanguage::Python]);

        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.max_output_size, 1024);
        assert_eq!(config.allowed_languages.len(), 1);
    }

    #[test]
    fn test_programming_language() {
        assert_eq!(ProgrammingLanguage::Python.as_str(), "python");
        assert_eq!(
            ProgrammingLanguage::from_str("python"),
            Some(ProgrammingLanguage::Python)
        );
        assert_eq!(
            ProgrammingLanguage::from_str("js"),
            Some(ProgrammingLanguage::JavaScript)
        );
        assert_eq!(ProgrammingLanguage::from_str("unknown"), None);
    }

    #[test]
    fn test_code_execution_tool_definition() {
        let tool = CodeExecutionTool::new();
        let def = <CodeExecutionTool as Tool<()>>::definition(&tool);
        assert_eq!(def.name, "code_execution");
        let required = def
            .parameters()
            .get("required")
            .and_then(|value| value.as_array())
            .unwrap();
        assert!(required
            .iter()
            .any(|value| value.as_str() == Some("language")));
        assert!(required.iter().any(|value| value.as_str() == Some("code")));
    }

    #[tokio::test]
    async fn test_code_execution_tool_call() {
        let tool = CodeExecutionTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(
                &ctx,
                serde_json::json!({
                    "language": "python",
                    "code": "print('hello')"
                }),
            )
            .await
            .unwrap();

        assert!(!result.is_error());
        let json = result.as_json().unwrap();
        assert!(json["success"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_code_execution_disallowed_language() {
        let tool = CodeExecutionTool::with_config(
            CodeExecutionConfig::new().allowed_languages(vec![ProgrammingLanguage::Python]),
        );
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(
                &ctx,
                serde_json::json!({
                    "language": "javascript",
                    "code": "console.log('hi')"
                }),
            )
            .await;

        assert!(matches!(result, Err(ToolError::ValidationFailed { .. })));
    }

    #[tokio::test]
    async fn test_code_execution_missing_code() {
        let tool = CodeExecutionTool::new();
        let ctx = RunContext::minimal("test");

        let result = tool
            .call(&ctx, serde_json::json!({"language": "python"}))
            .await;

        assert!(matches!(result, Err(ToolError::ValidationFailed { .. })));
    }

    #[test]
    fn test_execution_result() {
        let success = ExecutionResult {
            stdout: "output".to_string(),
            stderr: None,
            exit_code: 0,
            execution_time_ms: 100,
            timed_out: false,
        };
        assert!(success.is_success());

        let failure = ExecutionResult {
            stdout: "".to_string(),
            stderr: Some("error".to_string()),
            exit_code: 1,
            execution_time_ms: 100,
            timed_out: false,
        };
        assert!(!failure.is_success());

        let timeout = ExecutionResult {
            stdout: "".to_string(),
            stderr: None,
            exit_code: 0,
            execution_time_ms: 30000,
            timed_out: true,
        };
        assert!(!timeout.is_success());
    }
}
