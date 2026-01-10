//! Graph edge types.

use std::fmt;
use std::sync::Arc;

/// An edge between two nodes with an optional condition.
pub struct Edge<S> {
    /// Source node name.
    pub from: String,
    /// Target node name.
    pub to: String,
    /// Condition function.
    pub condition: Arc<dyn Fn(&S) -> bool + Send + Sync>,
    /// Edge label/description.
    pub label: Option<String>,
}

impl<S> Edge<S> {
    /// Create a new conditional edge.
    pub fn new<F>(from: impl Into<String>, to: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&S) -> bool + Send + Sync + 'static,
    {
        Self {
            from: from.into(),
            to: to.into(),
            condition: Arc::new(condition),
            label: None,
        }
    }

    /// Create an unconditional edge (always true).
    pub fn unconditional(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
            condition: Arc::new(|_| true),
            label: None,
        }
    }

    /// Add a label to the edge.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Check if the condition is satisfied.
    pub fn matches(&self, state: &S) -> bool {
        (self.condition)(state)
    }
}

impl<S> fmt::Debug for Edge<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Edge")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("label", &self.label)
            .finish()
    }
}

impl<S> Clone for Edge<S> {
    fn clone(&self) -> Self {
        Self {
            from: self.from.clone(),
            to: self.to.clone(),
            condition: Arc::clone(&self.condition),
            label: self.label.clone(),
        }
    }
}

/// Builder for creating edges.
#[derive(Debug)]
pub struct EdgeBuilder<S> {
    from: String,
    to: Option<String>,
    label: Option<String>,
    _phantom: std::marker::PhantomData<S>,
}

impl<S> EdgeBuilder<S> {
    /// Start building an edge from a node.
    pub fn from(name: impl Into<String>) -> Self {
        Self {
            from: name.into(),
            to: None,
            label: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the target node.
    pub fn to(mut self, name: impl Into<String>) -> Self {
        self.to = Some(name.into());
        self
    }

    /// Set the edge label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Build with a condition.
    pub fn when<F>(self, condition: F) -> Edge<S>
    where
        F: Fn(&S) -> bool + Send + Sync + 'static,
    {
        let to = self.to.expect("Target node required");
        let mut edge = Edge::new(self.from, to, condition);
        edge.label = self.label;
        edge
    }

    /// Build as unconditional.
    pub fn always(self) -> Edge<S> {
        let to = self.to.expect("Target node required");
        let mut edge = Edge::unconditional(self.from, to);
        edge.label = self.label;
        edge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestState {
        value: i32,
    }

    #[test]
    fn test_conditional_edge() {
        let edge = Edge::new("a", "b", |s: &TestState| s.value > 0);

        assert!(edge.matches(&TestState { value: 1 }));
        assert!(!edge.matches(&TestState { value: -1 }));
    }

    #[test]
    fn test_unconditional_edge() {
        let edge: Edge<TestState> = Edge::unconditional("a", "b");

        assert!(edge.matches(&TestState { value: 0 }));
        assert!(edge.matches(&TestState { value: -100 }));
    }

    #[test]
    fn test_edge_builder() {
        let edge = EdgeBuilder::<TestState>::from("start")
            .to("end")
            .label("always")
            .always();

        assert_eq!(edge.from, "start");
        assert_eq!(edge.to, "end");
        assert_eq!(edge.label, Some("always".to_string()));
    }

    #[test]
    fn test_edge_builder_conditional() {
        let edge = EdgeBuilder::<TestState>::from("a")
            .to("b")
            .when(|s| s.value == 42);

        assert!(edge.matches(&TestState { value: 42 }));
        assert!(!edge.matches(&TestState { value: 0 }));
    }

    #[test]
    fn test_edge_clone() {
        let edge = Edge::new("a", "b", |s: &TestState| s.value > 0);
        let cloned = edge.clone();

        assert_eq!(cloned.from, "a");
        assert_eq!(cloned.to, "b");
    }
}
