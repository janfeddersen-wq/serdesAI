//! Thinking and output controls of the Anthropic Messages API, shared by the
//! Anthropic and the Claude Code OAuth adapter.
//!
//! - [`ThinkingDisplay`]: `thinking.display` (`"summarized"`, `"omitted"`, or
//!   any other string passed through).
//! - [`Effort`]: `output_config.effort` (`"low"` ... `"max"`, or any other
//!   string passed through).
//! - [`BlockBinding`]: the opt-in `thinking.block_binding` of the beta
//!   [`THINKING_BINDING_BETA`].

use super::types::ThinkingConfig;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The `anthropic-beta` flag required by `thinking.block_binding`.
pub const THINKING_BINDING_BETA: &str = "thinking-binding-controls-2026-08-01";

/// The `anthropic-beta` flag the legacy `enabled` thinking form sends.
pub(crate) const INTERLEAVED_THINKING_BETA: &str = "interleaved-thinking-2025-05-14";

/// A wire string with known values and a pass-through for any other value.
macro_rules! wire_string_enum {
    (
        $(#[$meta:meta])*
        $name:ident { $($(#[$vmeta:meta])* $variant:ident => $wire:literal,)+ }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum $name {
            $($(#[$vmeta])* $variant,)+
            /// Any other value, sent verbatim.
            Other(String),
        }

        impl $name {
            /// The value as sent on the wire.
            #[must_use]
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $wire,)+
                    Self::Other(value) => value,
                }
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                match value {
                    $($wire => Self::$variant,)+
                    other => Self::Other(other.to_string()),
                }
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::from(value.as_str())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                Ok(Self::from(String::deserialize(deserializer)?))
            }
        }
    };
}

wire_string_enum! {
    /// How thinking is shown in the response (`thinking.display`).
    ThinkingDisplay {
        /// `"summarized"`: summarized thinking text.
        Summarized => "summarized",
        /// `"omitted"`: empty thinking text, signature only.
        Omitted => "omitted",
    }
}

wire_string_enum! {
    /// Output effort (`output_config.effort`).
    Effort {
        /// `"low"`.
        Low => "low",
        /// `"medium"`.
        Medium => "medium",
        /// `"high"`.
        High => "high",
        /// `"xhigh"`.
        XHigh => "xhigh",
        /// `"max"`.
        Max => "max",
    }
}

/// Output configuration of a request (`output_config`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
}

/// What the API does when a replayed thinking block no longer matches its
/// prefix (`thinking.block_binding.prefix_mismatch_behavior`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixMismatchBehavior {
    /// `"drop_block"`: drop the mismatching thinking block.
    DropBlock,
    /// `"error"`: reject the request.
    Error,
}

/// Opt-in thinking block binding (`thinking.block_binding`).
///
/// Requires the `anthropic-beta` flag [`THINKING_BINDING_BETA`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockBinding {
    /// Behaviour on a prefix mismatch.
    pub prefix_mismatch_behavior: PrefixMismatchBehavior,
}

impl BlockBinding {
    /// Create a block binding with the given mismatch behaviour.
    #[must_use]
    pub fn new(prefix_mismatch_behavior: PrefixMismatchBehavior) -> Self {
        Self {
            prefix_mismatch_behavior,
        }
    }
}

/// Which thinking form a model sends. The last builder call wins.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum ThinkingMode {
    /// No `thinking` object.
    #[default]
    Off,
    /// Legacy `{"type": "enabled", "budget_tokens": ...}`.
    Enabled {
        /// Budget tokens, if any.
        budget: Option<u64>,
    },
    /// `{"type": "adaptive"}`, no budget.
    Adaptive,
}

/// Thinking and output state of a model - the single source both adapters
/// build their request body and beta flags from.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ThinkingControls {
    /// Thinking form.
    pub(crate) mode: ThinkingMode,
    /// `thinking.display`, sent only while thinking is on.
    pub(crate) display: Option<ThinkingDisplay>,
    /// `output_config.effort`, sent whenever set.
    pub(crate) effort: Option<Effort>,
    /// `thinking.block_binding`, sent only while thinking is on.
    pub(crate) block_binding: Option<BlockBinding>,
}

impl ThinkingControls {
    /// The `thinking` object of the request body, if thinking is on.
    pub(crate) fn thinking_config(&self) -> Option<ThinkingConfig> {
        let config = match self.mode {
            ThinkingMode::Off => return None,
            ThinkingMode::Enabled { budget: Some(b) } => ThinkingConfig::with_budget(b),
            ThinkingMode::Enabled { budget: None } => ThinkingConfig::enabled(),
            ThinkingMode::Adaptive => ThinkingConfig::adaptive(),
        };
        Some(ThinkingConfig {
            display: self.display.clone(),
            block_binding: self.block_binding,
            ..config
        })
    }

    /// The `output_config` object of the request body, if effort is set.
    pub(crate) fn output_config(&self) -> Option<OutputConfig> {
        self.effort.clone().map(|effort| OutputConfig {
            effort: Some(effort),
        })
    }

    /// Whether the legacy `enabled` form is on, which sends
    /// [`INTERLEAVED_THINKING_BETA`]. Adaptive thinking interleaves without it.
    pub(crate) fn is_legacy_enabled(&self) -> bool {
        matches!(self.mode, ThinkingMode::Enabled { .. })
    }

    /// Whether `thinking.block_binding` is sent, which needs
    /// [`THINKING_BINDING_BETA`].
    pub(crate) fn sends_block_binding(&self) -> bool {
        self.block_binding.is_some() && self.mode != ThinkingMode::Off
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_strings_round_trip_and_pass_unknown_values_through() {
        assert_eq!(serde_json::to_value(Effort::XHigh).unwrap(), "xhigh");
        assert_eq!(Effort::from("max"), Effort::Max);
        assert_eq!(
            serde_json::from_value::<Effort>(serde_json::json!("ultra")).unwrap(),
            Effort::Other("ultra".into())
        );
        assert_eq!(
            serde_json::to_value(ThinkingDisplay::Other("full".into())).unwrap(),
            "full"
        );
        assert_eq!(ThinkingDisplay::from("omitted"), ThinkingDisplay::Omitted);
    }

    #[test]
    fn controls_build_the_thinking_object_only_while_thinking_is_on() {
        let mut controls = ThinkingControls {
            display: Some(ThinkingDisplay::Summarized),
            block_binding: Some(BlockBinding::new(PrefixMismatchBehavior::DropBlock)),
            ..ThinkingControls::default()
        };
        assert!(controls.thinking_config().is_none());
        assert!(!controls.sends_block_binding());

        controls.mode = ThinkingMode::Adaptive;
        assert_eq!(
            serde_json::to_value(controls.thinking_config().unwrap()).unwrap(),
            serde_json::json!({
                "type": "adaptive",
                "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"}
            })
        );
        assert!(controls.sends_block_binding());
        assert!(!controls.is_legacy_enabled());
    }
}
