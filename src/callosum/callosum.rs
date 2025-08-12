use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc::{Sender, Receiver, channel},
    Arc, RwLock,
};
use std::time::SystemTime;
use super::event::{CallosumEvent, EventEnvelope};

type Subscriber = Sender<Arc<EventEnvelope>>;

pub struct CorpusCallosum {
    subscribers: Arc<RwLock<HashMap<String, Subscriber>>>,
    seq: AtomicU64,
}

impl CorpusCallosum {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            seq: AtomicU64::new(1),
        }
    }

    /// Add/replace a subscriber by name. Returns its receiver.
    pub fn subscribe(&self, name: &str) -> Receiver<Arc<EventEnvelope>> {
        let (tx, rx) = channel();
        let mut subs = self.subscribers.write().expect("write lock");
        subs.insert(name.to_string(), tx);
        rx
    }

    /// Remove a subscriber by name (optional convenience).
    pub fn unsubscribe(&self, name: &str) {
        let mut subs = self.subscribers.write().expect("write lock");
        subs.remove(name);
    }

    /// Publish a fully-formed envelope (if you crafted one yourself).
    pub fn publish_envelope(&self, env: EventEnvelope) {
        let shared = Arc::new(env);
        let mut dead: Vec<String> = Vec::new();

        {
            let subs = self.subscribers.read().expect("read lock");
            for (name, sub) in subs.iter() {
                if sub.send(shared.clone()).is_err() {
                    dead.push(name.clone());
                }
            }
        }

        if !dead.is_empty() {
            let mut subs = self.subscribers.write().expect("write lock");
            for name in dead { subs.remove(&name); }
        }
    }

    /// Convenience: publish by parts. Adds id, ts, source, corr id.
    pub fn publish(&self, source: &'static str, correlation_id: Option<u64>, event: CallosumEvent) -> u64 {
        let id = self.seq.fetch_add(1, Ordering::Relaxed);
        let env = EventEnvelope {
            id,
            ts: SystemTime::now(),
            source,
            correlation_id,
            event,
        };
        self.publish_envelope(env);
        id
    }

    /// Convenience: publish a dynamic payload.
    pub fn publish_dynamic<T>(
        &self,
        source: &'static str,
        correlation_id: Option<u64>,
        payload: T,
    ) -> u64
    where
        T: crate::callosum::event::CloneableAny + 'static + std::clone::Clone, // path adjust if needed
    {
        self.publish(source, correlation_id, CallosumEvent::dynamic(payload))
    }

    /// Current number of subscribers (debug aid).
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.read().map(|m| m.len()).unwrap_or(0)
    }
}
// in callosum/callosum.rs
impl CorpusCallosum {
    pub fn publish_env(&self, env: EventEnvelope) -> u64 {
        self.publish(env.source, env.correlation_id, env.event)
    }
}