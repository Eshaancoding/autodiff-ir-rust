// contains the dependency tracker
use std::{collections::HashSet, sync::Mutex};

pub static DEP_TRACKER: Mutex<Option<HashSet<String>>> = Mutex::new(None);
pub static HARSH_DEP_LIST: Mutex<bool> = Mutex::new(false);

pub fn add_to_dep (id: String) {
    let mut guard = DEP_TRACKER.lock().unwrap();
    let dp = guard.as_mut().expect("Can't unpack dep tracker");
    dp.insert(id);
}

pub fn is_in_dep (id: String) -> bool {
    let mut guard = DEP_TRACKER.lock().unwrap();
    let dp = guard.as_mut().expect("Can't unpack dep tracker");
    dp.contains(&id)
}

pub fn ret_dep_list () -> HashSet<String> {
    let mut guard = DEP_TRACKER.lock().unwrap();
    let dp = guard.as_mut().expect("Can't unpack dep tracker");
    dp.clone()
}

pub fn set_harsh_dep_list () {
    let mut guard = HARSH_DEP_LIST.lock().unwrap();
    *guard = true; // set to true
}

pub fn is_harsh () -> bool {
    let guard = HARSH_DEP_LIST.lock().unwrap();
    guard.clone()
}