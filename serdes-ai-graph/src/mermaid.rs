//! Mermaid diagram generation for graphs.

use crate::graph::Graph;
use crate::state::GraphState;

/// Options for Mermaid diagram generation.
#[derive(Debug, Clone, Default)]
pub struct MermaidOptions {
    /// Diagram direction.
    pub direction: MermaidDirection,
    /// Theme name.
    pub theme: Option<String>,
    /// Include node descriptions.
    pub include_descriptions: bool,
    /// Use curved edges.
    pub curved_edges: bool,
}

impl MermaidOptions {
    /// Create new default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set direction.
    pub fn direction(mut self, dir: MermaidDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Set theme.
    pub fn theme(mut self, theme: impl Into<String>) -> Self {
        self.theme = Some(theme.into());
        self
    }

    /// Include descriptions.
    pub fn with_descriptions(mut self) -> Self {
        self.include_descriptions = true;
        self
    }

    /// Use curved edges.
    pub fn curved(mut self) -> Self {
        self.curved_edges = true;
        self
    }
}

/// Direction for Mermaid diagram.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MermaidDirection {
    /// Top to bottom.
    #[default]
    TopDown,
    /// Left to right.
    LeftRight,
    /// Bottom to top.
    BottomUp,
    /// Right to left.
    RightLeft,
}

impl MermaidDirection {
    fn as_str(&self) -> &'static str {
        match self {
            Self::TopDown => "TD",
            Self::LeftRight => "LR",
            Self::BottomUp => "BT",
            Self::RightLeft => "RL",
        }
    }
}

/// Generate a Mermaid diagram from a graph.
pub fn generate_mermaid<State, Deps, End>(
    graph: &Graph<State, Deps, End>,
    options: &MermaidOptions,
) -> String
where
    State: GraphState,
    Deps: Send + Sync + 'static,
    End: Send + Sync + 'static,
{
    let mut output = String::new();

    // Diagram header
    output.push_str("graph ");
    output.push_str(options.direction.as_str());
    output.push('\n');

    // Add theme if specified
    if let Some(ref theme) = options.theme {
        output.push_str(&format!("    %%{{init: {{'theme': '{}'}}}}%%\n", theme));
    }

    // Add nodes
    for (name, node) in &graph.nodes {
        let safe_id = sanitize_id(name);
        let label = if options.include_descriptions {
            format!("{}[\"{}\\n{}\"]", safe_id, name, node.node.type_name())
        } else {
            format!("{}[{}]", safe_id, name)
        };
        output.push_str(&format!("    {}\n", label));
    }

    output.push('\n');

    // Add edges
    for edge in graph.edges() {
        let from_id = sanitize_id(&edge.from);
        let to_id = sanitize_id(&edge.to);
        let arrow = if options.curved_edges { "-..->" } else { "-->" };

        if let Some(label) = &edge.label {
            output.push_str(&format!("    {} {}|{}| {}\n", from_id, arrow, label, to_id));
        } else {
            output.push_str(&format!("    {} {} {}\n", from_id, arrow, to_id));
        }
    }

    output
}

/// Generate a simple flowchart.
pub fn generate_flowchart(
    nodes: &[&str],
    edges: &[(&str, &str, Option<&str>)],
    options: &MermaidOptions,
) -> String {
    let mut output = String::new();

    // Header
    output.push_str("flowchart ");
    output.push_str(options.direction.as_str());
    output.push('\n');

    // Nodes
    for name in nodes {
        let safe_id = sanitize_id(name);
        output.push_str(&format!("    {}[{}]\n", safe_id, name));
    }

    output.push('\n');

    // Edges
    for (from, to, label) in edges {
        let from_id = sanitize_id(from);
        let to_id = sanitize_id(to);

        if let Some(lbl) = label {
            output.push_str(&format!("    {} -->|{}| {}\n", from_id, lbl, to_id));
        } else {
            output.push_str(&format!("    {} --> {}\n", from_id, to_id));
        }
    }

    output
}

/// Generate a state diagram.
pub fn generate_state_diagram(_states: &[&str], transitions: &[(&str, &str, &str)]) -> String {
    let mut output = String::new();

    output.push_str("stateDiagram-v2\n");

    for (from, to, event) in transitions {
        let from_id = if *from == "[*]" {
            "[*]".to_string()
        } else {
            sanitize_id(from)
        };
        let to_id = if *to == "[*]" {
            "[*]".to_string()
        } else {
            sanitize_id(to)
        };

        output.push_str(&format!("    {} --> {}: {}\n", from_id, to_id, event));
    }

    output
}

/// Sanitize a string for use as a Mermaid ID.
fn sanitize_id(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Builder for creating Mermaid diagrams.
#[derive(Debug, Default)]
pub struct MermaidBuilder {
    diagram_type: DiagramType,
    direction: MermaidDirection,
    nodes: Vec<NodeSpec>,
    edges: Vec<EdgeSpec>,
    subgraphs: Vec<SubgraphSpec>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
enum DiagramType {
    #[default]
    Flowchart,
    StateDiagram,
    Graph,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct NodeSpec {
    id: String,
    label: String,
    shape: NodeShape,
    style: Option<String>,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
enum NodeShape {
    #[default]
    Rectangle,
    RoundedRect,
    Stadium,
    Circle,
    Diamond,
    Hexagon,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EdgeSpec {
    from: String,
    to: String,
    label: Option<String>,
    style: EdgeStyle,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
enum EdgeStyle {
    #[default]
    Solid,
    Dashed,
    Thick,
}

#[derive(Debug, Clone)]
struct SubgraphSpec {
    id: String,
    label: String,
    nodes: Vec<String>,
}

impl MermaidBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set diagram type to flowchart.
    pub fn flowchart(mut self) -> Self {
        self.diagram_type = DiagramType::Flowchart;
        self
    }

    /// Set diagram type to state diagram.
    pub fn state_diagram(mut self) -> Self {
        self.diagram_type = DiagramType::StateDiagram;
        self
    }

    /// Set direction.
    pub fn direction(mut self, dir: MermaidDirection) -> Self {
        self.direction = dir;
        self
    }

    /// Add a node.
    pub fn node(mut self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.nodes.push(NodeSpec {
            id: id.into(),
            label: label.into(),
            shape: NodeShape::default(),
            style: None,
        });
        self
    }

    /// Add an edge.
    pub fn edge(
        mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        label: Option<&str>,
    ) -> Self {
        self.edges.push(EdgeSpec {
            from: from.into(),
            to: to.into(),
            label: label.map(String::from),
            style: EdgeStyle::default(),
        });
        self
    }

    /// Add a subgraph.
    pub fn subgraph(
        mut self,
        id: impl Into<String>,
        label: impl Into<String>,
        nodes: &[&str],
    ) -> Self {
        self.subgraphs.push(SubgraphSpec {
            id: id.into(),
            label: label.into(),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
        });
        self
    }

    /// Build the Mermaid diagram.
    pub fn build(self) -> String {
        let mut output = String::new();

        // Header
        match self.diagram_type {
            DiagramType::Flowchart => {
                output.push_str(&format!("flowchart {}\n", self.direction.as_str()));
            }
            DiagramType::StateDiagram => {
                output.push_str("stateDiagram-v2\n");
            }
            DiagramType::Graph => {
                output.push_str(&format!("graph {}\n", self.direction.as_str()));
            }
        }

        // Subgraphs
        for sg in &self.subgraphs {
            output.push_str(&format!("    subgraph {}[{}]\n", sg.id, sg.label));
            for node_id in &sg.nodes {
                output.push_str(&format!("        {}\n", node_id));
            }
            output.push_str("    end\n");
        }

        // Nodes
        for node in &self.nodes {
            let shape = match node.shape {
                NodeShape::Rectangle => format!("{}[{}]", node.id, node.label),
                NodeShape::RoundedRect => format!("{}({})", node.id, node.label),
                NodeShape::Stadium => format!("{}([{}])", node.id, node.label),
                NodeShape::Circle => format!("{}(({}))", node.id, node.label),
                NodeShape::Diamond => format!("{}{{{{{}}}}} ", node.id, node.label),
                NodeShape::Hexagon => format!("{}{{{{{{{}}}}}}} ", node.id, node.label),
            };
            output.push_str(&format!("    {}\n", shape));
        }

        output.push('\n');

        // Edges
        for edge in &self.edges {
            let arrow = match edge.style {
                EdgeStyle::Solid => "-->",
                EdgeStyle::Dashed => "-.->",
                EdgeStyle::Thick => "==>",
            };

            if let Some(ref label) = edge.label {
                output.push_str(&format!(
                    "    {} {}|{}| {}\n",
                    edge.from, arrow, label, edge.to
                ));
            } else {
                output.push_str(&format!("    {} {} {}\n", edge.from, arrow, edge.to));
            }
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_id() {
        assert_eq!(sanitize_id("hello world"), "hello_world");
        assert_eq!(sanitize_id("test-node"), "test_node");
        assert_eq!(sanitize_id("valid_id"), "valid_id");
    }

    #[test]
    fn test_flowchart() {
        let diagram = generate_flowchart(
            &["start", "process", "end"],
            &[("start", "process", None), ("process", "end", Some("done"))],
            &MermaidOptions::new(),
        );

        assert!(diagram.contains("flowchart TD"));
        assert!(diagram.contains("start[start]"));
        assert!(!diagram.contains("process --> end"));
        assert!(diagram.contains("|done|"));
    }

    #[test]
    fn test_mermaid_builder() {
        let diagram = MermaidBuilder::new()
            .flowchart()
            .direction(MermaidDirection::LeftRight)
            .node("a", "Start")
            .node("b", "End")
            .edge("a", "b", Some("next"))
            .build();

        assert!(diagram.contains("flowchart LR"));
        assert!(diagram.contains("a[Start]"));
        assert!(diagram.contains("b[End]"));
        assert!(diagram.contains("|next|"));
    }

    #[test]
    fn test_state_diagram() {
        let diagram = generate_state_diagram(
            &["idle", "running", "stopped"],
            &[
                ("[*]", "idle", "init"),
                ("idle", "running", "start"),
                ("running", "stopped", "stop"),
            ],
        );

        assert!(diagram.contains("stateDiagram-v2"));
        assert!(diagram.contains("[*] --> idle: init"));
    }

    #[test]
    fn test_options() {
        let opts = MermaidOptions::new()
            .direction(MermaidDirection::LeftRight)
            .theme("dark")
            .with_descriptions();

        assert_eq!(opts.direction, MermaidDirection::LeftRight);
        assert_eq!(opts.theme, Some("dark".to_string()));
        assert!(opts.include_descriptions);
    }
}
