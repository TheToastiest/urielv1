pub fn unified_score(
    sim: Option<f32>,
    recency: Option<f32>,
    drive_align: f32,
    source: &str,
) -> f32 {
    let base = match source {
        "smie"     => 0.9,
        "episodic" => 0.8,
        "concept"  => 1.0,
        "rows"     => 0.5,
        _          => 0.3,
    };

    let sim_w = sim.unwrap_or(0.0);
    let rec_w = recency.unwrap_or(0.0);

    let score =
        0.40 * base
            + 0.30 * sim_w
            + 0.20 * rec_w
            + 0.10 * drive_align;

    score.min(1.0)
}
