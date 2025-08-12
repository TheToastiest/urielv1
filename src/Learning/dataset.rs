use crate::tokenizer::tokenizer::Vocab;
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct EventRow {
    pub norm_input: String,
    pub selected_mem_text: String,
    pub salience: [f32;4],
    pub route: usize,
    pub cav: [f32;11],
}

pub struct JsonlDataset {
    rows: Vec<EventRow>,
    vocab: Vocab,
    pub max_len: usize,
}

impl JsonlDataset {
    pub fn load(path: &str, max_len: usize, vocab: Vocab) -> anyhow::Result<Self> {
        let f = std::fs::File::open(path)?;
        let rdr = std::io::BufRead::lines(std::io::BufReader::new(f));
        let mut rows = Vec::new();
        for line in rdr {
            rows.push(serde_json::from_str::<EventRow>(&line?)?);
        }
        Ok(Self { rows, vocab, max_len })
    }

    pub fn sample_batch(&self, bsz: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<[f32;4]>, Vec<usize>, Vec<[f32;11]>) {
        use rand::{seq::SliceRandom, thread_rng};
        let mut rng = thread_rng();
        let batch = self.rows.choose_multiple(&mut rng, bsz);
        let mut q_tokens = Vec::with_capacity(bsz);
        let mut p_tokens = Vec::with_capacity(bsz);
        let mut sals = Vec::with_capacity(bsz);
        let mut routes = Vec::with_capacity(bsz);
        let mut cavs = Vec::with_capacity(bsz);
        for r in batch {
            q_tokens.push(self.vocab.encode(&r.norm_input, self.max_len));
            p_tokens.push(self.vocab.encode(&r.selected_mem_text, self.max_len));
            sals.push(r.salience);
            routes.push(r.route);
            cavs.push(r.cav);
        }
        (q_tokens, p_tokens, sals, routes, cavs)
    }
}
