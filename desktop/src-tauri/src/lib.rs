mod engine;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            engine::setup(app.handle())?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            engine::get_state,
            engine::install_env,
            engine::download_model,
            engine::remove_model,
            engine::import_model,
            engine::start_service,
            engine::stop_service,
            engine::update_settings,
            engine::get_api_key,
            engine::open_path,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
