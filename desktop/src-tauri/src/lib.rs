// Launches the NoodleStockTracker Streamlit server as a background process
// (reusing it if already running) and embeds it in a native webview window,
// instead of opening Chrome in --app mode.

use std::fs::{self, OpenOptions};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use tauri::{WebviewUrl, WebviewWindowBuilder};

const APP_FILE: &str = "app.py";
const PORT: u16 = 8501;

/// Root of the NoodleStockTracker project. Resolved at compile time relative
/// to this crate (`desktop/src-tauri`, two levels under the project root)
/// rather than hardcoded, so a build works correctly no matter where the
/// repo was cloned — `CARGO_MANIFEST_DIR` reflects *the building machine's*
/// checkout location, not the original author's.
fn project_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // desktop/
        .and_then(Path::parent) // NoodleStockTracker/
        .expect("desktop/src-tauri should sit two levels under the project root")
        .to_path_buf()
}

fn port_is_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        Duration::from_millis(300),
    )
    .is_ok()
}

/// Resolves the `streamlit` binary via the user's login shell PATH, since
/// GUI-launched apps don't inherit the interactive shell environment. Not
/// hardcoded to a specific Python install, so this works for any Python
/// setup (framework build, Homebrew, pyenv, a venv activated in .zshrc, etc.)
/// as long as `streamlit` is installed and on PATH.
fn resolve_streamlit_bin() -> Option<String> {
    let output = Command::new("/bin/zsh")
        .args(["-lc", "command -v streamlit"])
        .output()
        .ok()?;
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

fn show_error_dialog(message: &str) {
    let script = format!(
        "display dialog \"{}\" buttons {{\"OK\"}} default button 1 with icon stop with title \"Noodle Stock Tracker\"",
        message.replace('\"', "'")
    );
    let _ = Command::new("osascript").args(["-e", &script]).spawn();
}

fn ensure_server_running() -> bool {
    if port_is_open(PORT) {
        return true;
    }

    let Some(streamlit) = resolve_streamlit_bin() else {
        show_error_dialog("Streamlit not found. Install it with:\\npip3 install streamlit");
        return false;
    };

    let project_dir = project_dir();
    if !project_dir.is_dir() {
        show_error_dialog(&format!("Project directory not found: {}", project_dir.display()));
        return false;
    }

    let log_dir = format!(
        "{}/Library/Logs/NoodleStockTracker",
        std::env::var("HOME").unwrap_or_default()
    );
    let _ = fs::create_dir_all(&log_dir);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{log_dir}/server.log"))
        .ok();
    let log_file_err = log_file.as_ref().and_then(|f| f.try_clone().ok());

    let spawn_result = Command::new(&streamlit)
        .args([
            "run",
            APP_FILE,
            "--server.port",
            &PORT.to_string(),
            "--server.headless",
            "true",
            "--server.runOnSave",
            "false",
            "--browser.gatherUsageStats",
            "false",
        ])
        .current_dir(&project_dir)
        .stdout(log_file.map(Stdio::from).unwrap_or_else(Stdio::null))
        .stderr(log_file_err.map(Stdio::from).unwrap_or_else(Stdio::null))
        .spawn();

    if spawn_result.is_err() {
        show_error_dialog("Failed to start the Streamlit server.");
        return false;
    }

    // Wait up to ~20s for the server to accept connections.
    for _ in 0..40 {
        if port_is_open(PORT) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    show_error_dialog("Timed out waiting for the Streamlit server to start.");
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let ready = ensure_server_running();
            let url = if ready {
                format!("http://localhost:{PORT}")
            } else {
                "about:blank".to_string()
            };

            WebviewWindowBuilder::new(
                app,
                "main",
                WebviewUrl::External(url.parse().expect("valid url")),
            )
            .title("Noodle Stock Tracker")
            .inner_size(1600.0, 1000.0)
            .min_inner_size(960.0, 640.0)
            .build()?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}