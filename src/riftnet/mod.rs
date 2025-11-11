use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::os::raw::{c_char, c_uchar};
use std::sync::{Arc, Mutex, mpsc::{Sender, Receiver, channel}};
use std::{ptr, time::Duration};
use std::thread;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub enum RiftResult {
    Success = 0,
    ErrorGeneric = -1,
    ErrorInvalidHandle = -2,
    ErrorInvalidParameter = -3,
    ErrorSocketCreationFailed = -4,
    ErrorSocketBindFailed = -5,
    ErrorConnectionFailed = -6,
    ErrorSendFailed = -7,
    ErrorIocpCreationFailed = -8,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RiftEventType {
    ServerStart = 0,
    ServerStop = 1,
    ClientConnected = 2,
    ClientDisconnected = 3,
    PacketReceived = 4,
}

pub type RiftClientId = u64;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RiftPacket {
    pub data: *const c_uchar,
    pub size: usize,
    pub sender_id: RiftClientId,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RiftEvent {
    pub r#type: RiftEventType,
    pub data: RiftEventData,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union RiftEventData {
    pub packet: RiftPacket,
    pub client_id: RiftClientId,
}

pub type RiftEventCallback = extern "C" fn(event: *const RiftEvent, user: *mut c_void);

#[repr(C)]
pub struct RiftClientConfig {
    pub event_callback: RiftEventCallback,
    pub user_data: *mut c_void,
}

#[repr(C)]
pub struct RiftServerConfig {
    pub host_address: *const c_char,
    pub port: u16,
    pub event_callback: RiftEventCallback,
    pub user_data: *mut c_void,
}

/* opaque handles */
#[repr(C)] pub struct RiftClient;
#[repr(C)] pub struct RiftServer;

type FnAbi          = unsafe extern "C" fn() -> u32;
type FnCliCreate    = unsafe extern "C" fn(cfg: *const RiftClientConfig) -> *mut RiftClient;
type FnCliDestroy   = unsafe extern "C" fn(cli: *mut RiftClient);
type FnCliConnect   = unsafe extern "C" fn(cli: *mut RiftClient, host: *const c_char, port: u16) -> i32;
type FnCliDisconnect= unsafe extern "C" fn(cli: *mut RiftClient);
type FnCliSend      = unsafe extern "C" fn(cli: *mut RiftClient, data: *const c_uchar, size: usize) -> i32;
type FnCliPoll      = unsafe extern "C" fn(cli: *mut RiftClient, max_millis: u32) -> i32;

pub struct RiftClientDyn {
    lib: Library,
    get_abi: FnAbi,
    cli_create: FnCliCreate,
    cli_destroy: FnCliDestroy,
    cli_connect: FnCliConnect,
    cli_disconnect: FnCliDisconnect,
    cli_send: FnCliSend,
    cli_poll: FnCliPoll,

    // runtime
    handle: *mut RiftClient,
    tx_for_cb: Arc<Mutex<Option<Sender<RiftEventOwned>>>>,
    poll_thread: Option<thread::JoinHandle<()>>,
}

#[derive(Clone, Debug)]
pub struct RiftEventOwned {
    pub kind: RiftEventType,
    pub sender_id: Option<RiftClientId>,
    pub bytes: Option<Vec<u8>>, // present for PacketReceived
}

// trampoline: forward C callback to mpsc
extern "C" fn cb_trampoline(ev: *const RiftEvent, user: *mut c_void) {
    if ev.is_null() || user.is_null() { return; }
    let tx_ptr = user as *mut Arc<Mutex<Option<Sender<RiftEventOwned>>>>;
    let tx_arc = unsafe { &*tx_ptr };
    let maybe_tx = tx_arc.lock().ok().and_then(|g| g.clone());
    if let Some(tx) = maybe_tx {
        let e = unsafe { &*ev };
        let owned = match e.r#type {
            RiftEventType::PacketReceived => {
                let pkt = unsafe { e.data.packet };
                let mut v = Vec::<u8>::with_capacity(pkt.size);
                if pkt.size > 0 && !pkt.data.is_null() {
                    unsafe { v.extend_from_slice(std::slice::from_raw_parts(pkt.data, pkt.size)); }
                }
                RiftEventOwned { kind: RiftEventType::PacketReceived, sender_id: Some(pkt.sender_id), bytes: Some(v) }
            }
            RiftEventType::ClientConnected => {
                let id = unsafe { e.data.client_id };
                RiftEventOwned { kind: RiftEventType::ClientConnected, sender_id: Some(id), bytes: None }
            }
            RiftEventType::ClientDisconnected => {
                let id = unsafe { e.data.client_id };
                RiftEventOwned { kind: RiftEventType::ClientDisconnected, sender_id: Some(id), bytes: None }
            }
            RiftEventType::ServerStart | RiftEventType::ServerStop => {
                RiftEventOwned { kind: e.r#type, sender_id: None, bytes: None }
            }
        };
        let _ = tx.send(owned);
    }
}

impl RiftClientDyn {
    pub fn load(dll_path: &str) -> anyhow::Result<Self> {
        unsafe {
            let lib = Library::new(dll_path)?;

            // Copy function pointers out in a short inner scope.
            let (
                fn_get_abi,
                fn_cli_create,
                fn_cli_destroy,
                fn_cli_connect,
                fn_cli_disconnect,
                fn_cli_send,
                fn_cli_poll,
            ) = {
                let get_abi: Symbol<FnAbi>            = lib.get(b"rift_get_abi_version\0")?;
                let cli_create: Symbol<FnCliCreate>   = lib.get(b"rift_client_create\0")?;
                let cli_destroy: Symbol<FnCliDestroy> = lib.get(b"rift_client_destroy\0")?;
                let cli_connect: Symbol<FnCliConnect> = lib.get(b"rift_client_connect\0")?;
                let cli_disconnect: Symbol<FnCliDisconnect> = lib.get(b"rift_client_disconnect\0")?;
                let cli_send: Symbol<FnCliSend>       = lib.get(b"rift_client_send\0")?;
                let cli_poll: Symbol<FnCliPoll>       = lib.get(b"rift_client_poll\0")?;

                (
                    *get_abi,
                    *cli_create,
                    *cli_destroy,
                    *cli_connect,
                    *cli_disconnect,
                    *cli_send,
                    *cli_poll,
                )
            }; // <- Symbols dropped here; now we can move `lib`

            let abi = fn_get_abi();
            if (abi >> 16) != 0x0001 {
                anyhow::bail!("RiftNet ABI mismatch: got {:08x}", abi);
            }

            Ok(Self {
                lib,
                get_abi: fn_get_abi,
                cli_create: fn_cli_create,
                cli_destroy: fn_cli_destroy,
                cli_connect: fn_cli_connect,
                cli_disconnect: fn_cli_disconnect,
                cli_send: fn_cli_send,
                cli_poll: fn_cli_poll,

                handle: std::ptr::null_mut(),
                tx_for_cb: Arc::new(Mutex::new(None)),
                poll_thread: None,
            })
        }
    }


    pub fn start(&mut self, host: &str, port: u16) -> anyhow::Result<Receiver<RiftEventOwned>> {
        let (tx, rx) = std::sync::mpsc::channel::<RiftEventOwned>();
        *self.tx_for_cb.lock().unwrap() = Some(tx);

        let user_ptr = &self.tx_for_cb as *const _ as *mut std::ffi::c_void;
        let cfg = RiftClientConfig { event_callback: cb_trampoline, user_data: user_ptr };

        unsafe {
            self.handle = (self.cli_create)(&cfg as *const _);
            if self.handle.is_null() {
                anyhow::bail!("rift_client_create failed");
            }
        }

        let host_c = std::ffi::CString::new(host)?;
        let rc = unsafe { (self.cli_connect)(self.handle, host_c.as_ptr(), port) };
        if rc != 0 {
            anyhow::bail!("rift_client_connect rc={}", rc);
        }

        // ---- Spawn poll loop (pass handle as usize; copy fn pointer) ----
        let handle_val: usize = self.handle as usize;
        let poll_fn = self.cli_poll;
        self.poll_thread = Some(std::thread::spawn(move || {
            let handle = handle_val as *mut RiftClient;
            loop {
                // drive callbacks; 50ms max wait inside the DLL
                let rc = unsafe { poll_fn(handle, 50) };
                // if the DLL signals invalid handle (or any fatal negative), stop
                if rc == (RiftResult::ErrorInvalidHandle as i32) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }));

        Ok(rx)
    }

    pub fn shutdown(mut self) {
        unsafe {
            if !self.handle.is_null() {
                (self.cli_disconnect)(self.handle);
                (self.cli_destroy)(self.handle);
                self.handle = std::ptr::null_mut();
            }
        }
        if let Some(h) = self.poll_thread.take() {
            let _ = h.join();
        }
    }


    pub fn send_str(&self, s: &str) -> anyhow::Result<()> {
        if self.handle.is_null() { anyhow::bail!("client not started"); }
        let bytes = s.as_bytes();
        let rc = unsafe { (self.cli_send)(self.handle, bytes.as_ptr(), bytes.len()) };
        if rc != 0 { anyhow::bail!("rift_client_send rc={}", rc); }
        Ok(())
    }
    
}
