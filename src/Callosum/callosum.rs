// src/URIEL_Callosum/callosum.rs

use std::collections::HashMap;
use std::sync::{mpsc::{Sender, Receiver, channel}, Arc, RwLock};

use super::event::CallosumEvent;

type Subscriber = Sender<Arc<CallosumEvent>>;

pub struct CorpusCallosum {
    subscribers: Arc<RwLock<HashMap<String, Subscriber>>>,
}

impl CorpusCallosum {
    pub fn new() -> Self {
        CorpusCallosum {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn subscribe(&self, name: &str) -> Receiver<Arc<CallosumEvent>> {
        let (tx, rx) = channel();
        let mut subs = self.subscribers.write().expect("Failed to acquire write lock");
        subs.insert(name.to_string(), tx);
        rx
    }

    pub fn publish(&self, event: CallosumEvent) {
        let shared_event = Arc::new(event);
        let subs = self.subscribers.read().expect("Failed to acquire read lock");

        for (_name, sub) in subs.iter() {
            if let Err(e) = sub.send(shared_event.clone()) {
                eprintln!("Failed to send event to subscriber: {}", e);
            }
        }
    }
}
