// src/rift/mod.rs
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::mpsc::{Sender, channel, Receiver};
use std::sync::{Arc, Mutex};
use libloading::{Library, Symbol};

type FnInit = unsafe extern "C" fn(cfg: *const c_char) -> i32;
type FnSend = unsafe extern "C" fn(topic: *const c_char, payload: *const c_char) -> i32;
type FnSetCb = unsafe extern "C" fn(cb: extern "C" fn(*const c_char, *const c_char, *mut c_void), user: *mut c_void);
type FnShutdown = unsafe extern "C" fn();

pub struct RiftNet {
    _lib: Library, // keep alive
    init: FnInit,
    send: FnSend,
    set_cb: FnSetCb,
    shutdown: FnShutdown,
    tx_for_cb: Arc<Mutex<Option<Sender<(String,String)>>>>,
}

unsafe extern "C" fn cb_trampoline(topic: *const c_char, payload: *const c_char, user: *mut c_void) {
    if user.is_null() { return; }
    let tx_ptr = user as *mut Arc<Mutex<Option<Sender<(String,String)>>>>;
    let tx_arc = &*tx_ptr;
    let maybe_tx = tx_arc.lock().ok().and_then(|g| g.clone());
    if let Some(tx) = maybe_tx {
        let t = CStr::from_ptr(topic).to_string_lossy().into_owned();
        let p = CStr::from_ptr(payload).to_string_lossy().into_owned();
        let _ = tx.send((t, p));
    }
}

impl RiftNet {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let lib = unsafe { Library::new(path)? };
        let init: Symbol<FnInit>     = unsafe { lib.get(b"Rift_Init\0")? };
        let send: Symbol<FnSend>     = unsafe { lib.get(b"Rift_Send\0")? };
        let set_cb: Symbol<FnSetCb>  = unsafe { lib.get(b"Rift_SetCallback\0")? };
        let shutdown: Symbol<FnShutdown> = unsafe { lib.get(b"Rift_Shutdown\0")? };
        Ok(Self {
            _lib: lib,
            init: *init,
            send: *send,
            set_cb: *set_cb,
            shutdown: *shutdown,
            tx_for_cb: Arc::new(Mutex::new(None)),
        })
    }

    pub fn init(&self, config_json: &str) -> anyhow::Result<()> {
        let c = CString::new(config_json)?;
        let rc = unsafe { (self.init)(c.as_ptr()) };
        if rc != 0 { anyhow::bail!("Rift_Init rc={}", rc); }
        Ok(())
    }

    pub fn set_callback(&self) -> Receiver<(String,String)> {
        let (tx, rx) = channel::<(String,String)>();
        *self.tx_for_cb.lock().unwrap() = Some(tx);
        // pass an Arc pointer as user data; we keep self alive during runtime
        let user = &self.tx_for_cb as *const _ as *mut c_void;
        unsafe { (self.set_cb)(cb_trampoline, user); }
        rx
    }

    pub fn send(&self, topic: &str, payload: &str) -> anyhow::Result<()> {
        let t = CString::new(topic)?;
        let p = CString::new(payload)?;
        let rc = unsafe { (self.send)(t.as_ptr(), p.as_ptr()) };
        if rc != 0 { anyhow::bail!("Rift_Send rc={}", rc); }
        Ok(())
    }

    pub fn shutdown(&self) {
        unsafe { (self.shutdown)(); }
    }
}
