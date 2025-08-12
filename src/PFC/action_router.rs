use crate::PFC::cortex::CortexAction;

/// Context for routing actions (dry-run vs real, origin, etc.)
#[derive(Debug, Clone, Copy, Default)]
pub struct ActionContext {
    pub dry_run: bool,
    pub source: &'static str, // e.g., "ui", "test", "daemon"
}

/// New: context-aware trigger
pub fn trigger_action_with(ctx: &ActionContext, action: &CortexAction) {
    let p = if ctx.dry_run { "[DRY] " } else { "" };

    match action {
        CortexAction::ScanItem => {
            println!("{p}[ACTION] Scanning item... (src={})", ctx.source);
            // if !ctx.dry_run { /* TODO: real impl */ }
        }
        CortexAction::ScanEnvironment => {
            println!("{p}[ACTION] Scanning environment... (src={})", ctx.source);
            // if !ctx.dry_run { /* TODO: real impl */ }
        }
        CortexAction::SearchInternet(query) => {
            if ctx.dry_run {
                println!("{p}[ACTION] Performing web search for: {} (src={})", query, ctx.source);
                // Optional: write a trace memory instead of actually calling out:
                // let _ = crate::Hippocampus::ingest(
                //     "action_trace".into(),
                //     format!("DRY WebSearch: {}", query),
                //     6 * 3600,
                //     "dry".into(),
                // );
            } else {
                println!("[ACTION] (REAL) Web search: {} (src={})", query, ctx.source);
                // TODO: real search call
            }
        }
        CortexAction::RunSubmodule(name) => {
            if ctx.dry_run {
                println!("{p}[ACTION] Running submodule: {} (src={})", name, ctx.source);
            } else {
                println!("[ACTION] (REAL) Run submodule: {} (src={})", name, ctx.source);
                // TODO: real submodule call
            }
        }
    }
}

/// Back-compat shim: defaults to DRY mode with an unspecified source.
/// Prefer `trigger_action_with`.
#[allow(dead_code)]
pub fn trigger_action(action: &CortexAction) {
    let ctx = ActionContext { dry_run: true, source: "unspecified" };
    trigger_action_with(&ctx, action);
}
